// WTinyLfuCache is a W-TinyLFU cache (the admission and eviction policy used by Caffeine), with all values shared,
// which allows each key value pair to be read from multiple requests. It exposes the same interfaces as
// `SharedValueLruCache`.
//
// Layout:
// - A small admission window (1% of capacity) managed as LRU, which absorbs bursts of new entries.
// - A main region managed as segmented LRU: a probation segment and a protected segment (80% of the main region).
// - A count-min sketch with 4-bit counters, which estimates access frequency for keys, including keys not in cache.
//   Counters are halved periodically so that frequency estimates age.
//
// When the window overflows, its LRU entry becomes an admission candidate for the main region. If the cache is over
// capacity, the candidate competes with the probation segment's LRU entry (the victim), and whichever has the lower
// estimated frequency is evicted. So a one-pass scan over cold data only churns the window and probation segment, and
// does not flush frequently accessed entries out of the cache.
//
// Frequency is recorded on every lookup (hit or miss), so a block that is read, evicted and read again accumulates
// frequency even when it's not in cache.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <list>
#include <utility>

#include "duckdb/common/helper.hpp"
#include "duckdb/common/map.hpp"
#include "duckdb/common/vector.hpp"
#include "map_utils.hpp"
#include "mutex.hpp"
#include "thread_annotation.hpp"
#include "time_utils.hpp"

namespace duckdb {

// Count-min sketch with 4-bit counters, 16 counters packed in each 64-bit word, and 4 hash functions.
class TinyLfuFrequencySketch {
public:
	// @param expected_entries: Expected number of entries in the cache, which decides sketch width and sample size.
	explicit TinyLfuFrequencySketch(size_t expected_entries) {
		size_t width = 8;
		while (width < expected_entries) {
			width <<= 1;
		}
		table.assign(width, 0);
		table_mask = width - 1;
		sample_size = 10 * (expected_entries == 0 ? 1 : expected_entries);
	}

	// Get the estimated access frequency for [hash], which is in [0, 15].
	uint32_t Frequency(uint64_t hash) const {
		uint32_t freq = MAX_COUNTER;
		for (uint32_t depth = 0; depth < DEPTH; ++depth) {
			const uint64_t idx = CounterIndex(hash, depth);
			freq = std::min(freq, static_cast<uint32_t>((table[idx >> 4] >> ((idx & 15) << 2)) & MAX_COUNTER));
		}
		return freq;
	}

	// Record one access for [hash].
	void Increment(uint64_t hash) {
		bool added = false;
		for (uint32_t depth = 0; depth < DEPTH; ++depth) {
			const uint64_t idx = CounterIndex(hash, depth);
			const uint64_t shift = (idx & 15) << 2;
			uint64_t &word = table[idx >> 4];
			if (((word >> shift) & MAX_COUNTER) != MAX_COUNTER) {
				word += (uint64_t {1} << shift);
				added = true;
			}
		}
		if (added && ++additions >= sample_size) {
			Reset();
		}
	}

	void Clear() {
		std::fill(table.begin(), table.end(), 0);
		additions = 0;
	}

private:
	static constexpr uint32_t DEPTH = 4;
	static constexpr uint32_t MAX_COUNTER = 15;

	// Halve all counters, so that old popularity fades out.
	void Reset() {
		for (auto &word : table) {
			word = (word >> 1) & 0x7777777777777777ULL;
		}
		additions /= 2;
	}

	// Index of the 4-bit counter for [hash] at row [depth]; each row has a different seed.
	uint64_t CounterIndex(uint64_t hash, uint32_t depth) const {
		static constexpr std::array<uint64_t, DEPTH> SEEDS {
		    {0xc3a5c85c97cb3127ULL, 0xb492b66fbe98f273ULL, 0x9ae16a3b2f90404fULL, 0xcbf29ce484222325ULL}};
		const uint64_t h = Mix(hash + SEEDS[depth]);
		// Each row spans the whole table: pick a word, then a counter within the word.
		return ((h & table_mask) << 4) | (h >> 60);
	}

	// splitmix64 finalizer.
	static uint64_t Mix(uint64_t x) {
		x ^= x >> 30;
		x *= 0xbf58476d1ce4e5b9ULL;
		x ^= x >> 27;
		x *= 0x94d049bb133111ebULL;
		x ^= x >> 31;
		return x;
	}

	vector<uint64_t> table;
	uint64_t table_mask = 0;
	// Number of increments since the last reset; counters are halved once it reaches [sample_size].
	size_t additions = 0;
	size_t sample_size = 0;
};

template <typename Key, typename Val, typename KeyCompare = std::less<Key>, typename KeyHash = std::hash<Key>>
class WTinyLfuCache {
public:
	using key_type = Key;
	using mapped_type = shared_ptr<Val>;
	using value_type = pair<const Key, shared_ptr<Val>>;
	using key_compare = KeyCompare;
	using hasher = KeyHash;

	// @param max_entries_p: A `max_entries` of 0 means that there is no limit on the number of entries in the cache.
	// @param timeout_millisec_p: Timeout in milliseconds for entries, exceeding which invalidates the cache entries; 0
	// means no timeout.
	WTinyLfuCache(size_t max_entries_p, uint64_t timeout_millisec_p)
	    : max_entries(max_entries_p), timeout_millisec(timeout_millisec_p),
	      window_capacity(max_entries_p == 0 ? 0 : std::max<size_t>(1, max_entries_p / 100)),
	      protected_capacity((max_entries_p - window_capacity) * 8 / 10), sketch(max_entries_p) {
	}

	// Disable copy and move.
	WTinyLfuCache(const WTinyLfuCache &) = delete;
	WTinyLfuCache &operator=(const WTinyLfuCache &) = delete;

	~WTinyLfuCache() = default;

	// Insert `value` with key `key`. This will replace any previous entry with the same key.
	void Put(Key key, shared_ptr<Val> value) {
		auto existing = entry_map.find(key);
		if (existing != entry_map.end()) {
			existing->second.value = std::move(value);
			existing->second.timestamp = static_cast<uint64_t>(GetSteadyNowMilliSecSinceEpoch());
			OnAccess(existing->second);
			return;
		}

		// New entries always land in the admission window.
		window_list.emplace_front(std::move(key));
		Entry new_entry {
		    .value = std::move(value),
		    .timestamp = static_cast<uint64_t>(GetSteadyNowMilliSecSinceEpoch()),
		    .segment = Segment::kWindow,
		    .list_iterator = window_list.begin(),
		};
		auto key_cref = std::cref(window_list.front());
		entry_map[key_cref] = std::move(new_entry);

		if (max_entries == 0) {
			return;
		}
		EvictIfNeeded();
	}

	// Delete the entry with key `key`. Return true if the entry was found for `key`, false if the entry was not found.
	// In both cases, there is no entry with key `key` existed after the call.
	bool Delete(const Key &key) {
		auto it = entry_map.find(key);
		if (it == entry_map.end()) {
			return false;
		}
		DeleteImpl(it);
		return true;
	}

	// Look up the entry with key `key`, and record the access for frequency estimation.
	// Return nullptr if `key` doesn't exist in cache.
	shared_ptr<Val> Get(const Key &key) {
		if (max_entries > 0) {
			sketch.Increment(KeyHash {}(key));
		}

		const auto entry_map_iter = entry_map.find(key);
		if (entry_map_iter == entry_map.end()) {
			return nullptr;
		}

		// Check whether found cache entry is expired or not.
		if (timeout_millisec > 0) {
			const auto now = GetSteadyNowMilliSecSinceEpoch();
			if (now - entry_map_iter->second.timestamp > timeout_millisec) {
				DeleteImpl(entry_map_iter);
				return nullptr;
			}
		}

		OnAccess(entry_map_iter->second);
		return entry_map_iter->second.value;
	}

	// Clear the cache, including frequency history.
	void Clear() {
		entry_map.clear();
		window_list.clear();
		probation_list.clear();
		protected_list.clear();
		sketch.Clear();
	}

	// Clear cache entries that match [key_filter] starting from [start_key] inclusively.
	// It ends at the first non-matched entry.
	template <typename KeyFilter>
	void Clear(const Key &start_key, KeyFilter &&key_filter) {
		vector<Key> keys_to_delete;
		for (auto iter = entry_map.lower_bound(start_key); iter != entry_map.end(); ++iter) {
			const Key &key = iter->first.get();
			if (!key_filter(key)) {
				break;
			}
			keys_to_delete.emplace_back(key);
		}
		for (const auto &key : keys_to_delete) {
			Delete(key);
		}
	}

	// Accessors for cache parameters.
	size_t MaxEntries() const {
		return max_entries;
	}

	// Get all keys inside of the cache; the order of keys returned is not deterministic.
	vector<Key> Keys() const {
		vector<Key> keys;
		keys.reserve(entry_map.size());
		for (const auto &[key_ref, _] : entry_map) {
			keys.emplace_back(key_ref.get());
		}
		return keys;
	}

	// Transfer all non-expired entries out of the cache; postcondition: empty cache. Order of pairs is unspecified.
	vector<pair<Key, shared_ptr<Val>>> Take() {
		vector<pair<Key, shared_ptr<Val>>> result;
		result.reserve(entry_map.size());
		while (!entry_map.empty()) {
			auto it = entry_map.begin();
			if (timeout_millisec > 0) {
				const auto now = GetSteadyNowMilliSecSinceEpoch();
				if (now - it->second.timestamp > timeout_millisec) {
					DeleteImpl(it);
					continue;
				}
			}
			Key key = it->first.get();
			shared_ptr<Val> val = std::move(it->second.value);
			DeleteImpl(it);
			result.emplace_back(std::move(key), std::move(val));
		}
		ALWAYS_ASSERT(window_list.empty());
		ALWAYS_ASSERT(probation_list.empty());
		ALWAYS_ASSERT(protected_list.empty());
		return result;
	}

	// Number of entries in each segment, exposed for testing.
	size_t WindowSize() const {
		return window_list.size();
	}
	size_t ProbationSize() const {
		return probation_list.size();
	}
	size_t ProtectedSize() const {
		return protected_list.size();
	}

private:
	enum class Segment : uint8_t {
		kWindow,
		kProbation,
		kProtected,
	};

	struct Entry {
		// The entry's value.
		shared_ptr<Val> value;

		// Steady clock timestamp when current entry was inserted into cache.
		// 1. It's not updated at later accesses.
		// 2. It's updated at replace update operations.
		uint64_t timestamp;

		// Which segment the entry currently lives in.
		Segment segment;

		// A list iterator pointing to the entry's position in its segment list.
		typename std::list<Key>::iterator list_iterator;
	};

	using KeyConstRef = std::reference_wrapper<const Key>;
	using EntryMap = map<KeyConstRef, Entry, RefLess<KeyCompare>>;

	std::list<Key> &SegmentList(Segment segment) {
		switch (segment) {
		case Segment::kWindow:
			return window_list;
		case Segment::kProbation:
			return probation_list;
		default:
			return protected_list;
		}
	}

	// Move [entry] to the front (most recently used end) of [segment]. List splicing keeps the key's address stable,
	// so references held by [entry_map] stay valid.
	void MoveToFront(Entry &entry, Segment segment) {
		auto &src = SegmentList(entry.segment);
		auto &dst = SegmentList(segment);
		dst.splice(dst.begin(), src, entry.list_iterator);
		entry.segment = segment;
	}

	// Update recency for a hit entry; a hit in probation promotes the entry to the protected segment.
	void OnAccess(Entry &entry) {
		if (max_entries == 0) {
			return;
		}
		if (entry.segment != Segment::kProbation) {
			MoveToFront(entry, entry.segment);
			return;
		}
		MoveToFront(entry, Segment::kProtected);
		// Protected segment overflows, demote its LRU entry back to probation.
		while (protected_list.size() > protected_capacity) {
			auto &demoted = entry_map.find(protected_list.back())->second;
			MoveToFront(demoted, Segment::kProbation);
		}
	}

	// Move window overflow into probation as admission candidates, then evict until the cache fits its capacity.
	void EvictIfNeeded() {
		// Candidates are moved to probation front in order, so the oldest candidate sits right after the newer ones.
		size_t candidate_count = 0;
		while (window_list.size() > window_capacity) {
			auto &entry = entry_map.find(window_list.back())->second;
			MoveToFront(entry, Segment::kProbation);
			++candidate_count;
		}

		while (entry_map.size() > max_entries) {
			// Pick the victim from the least recently used end of the main region.
			std::list<Key> *victim_list = !probation_list.empty()   ? &probation_list
			                              : !protected_list.empty() ? &protected_list
			                                                        : &window_list;
			auto victim_iter = entry_map.find(victim_list->back());

			// Every candidate already competed (or none was produced), evict the victim directly.
			if (candidate_count == 0 || victim_list != &probation_list) {
				DeleteImpl(victim_iter);
				continue;
			}

			// The oldest pending candidate is at position [candidate_count - 1] from probation front.
			auto candidate_key_iter = probation_list.begin();
			std::advance(candidate_key_iter, candidate_count - 1);
			--candidate_count;
			auto candidate_iter = entry_map.find(*candidate_key_iter);
			if (candidate_iter == victim_iter) {
				DeleteImpl(victim_iter);
				continue;
			}

			// Admit the candidate only if it's accessed more frequently than the victim; ties favor the incumbent.
			const auto candidate_freq = sketch.Frequency(KeyHash {}(candidate_iter->first.get()));
			const auto victim_freq = sketch.Frequency(KeyHash {}(victim_iter->first.get()));
			if (candidate_freq > victim_freq) {
				DeleteImpl(victim_iter);
			} else {
				DeleteImpl(candidate_iter);
			}
		}
	}

	// Delete key-value pairs indicated by the given entry map iterator [iter] from cache.
	void DeleteImpl(typename EntryMap::iterator iter) {
		auto &list = SegmentList(iter->second.segment);
		const auto list_iter = iter->second.list_iterator;
		entry_map.erase(iter);
		list.erase(list_iter);
	}

	// The maximum number of entries in the cache. A value of 0 means there is no limit on entry count.
	const size_t max_entries;

	// The timeout in seconds for cache entries; entries with exceeding timeout would be invalidated.
	const uint64_t timeout_millisec;

	// Capacity of the admission window and protected segment; probation takes the rest of the cache.
	const size_t window_capacity;
	const size_t protected_capacity;

	TinyLfuFrequencySketch sketch;

	// All keys are stored as refernce (`std::reference_wrapper`), and the ownership lies in the segment lists.
	EntryMap entry_map;

	// Segment lists. The front of each list identifies the most recently accessed entry.
	std::list<Key> window_list;
	std::list<Key> probation_list;
	std::list<Key> protected_list;
};

// Thread-safe implementation.
template <typename Key, typename Val, typename KeyCompare = std::less<Key>, typename KeyHash = std::hash<Key>>
class ThreadSafeWTinyLfuCache {
public:
	using cache_impl = WTinyLfuCache<Key, Val, KeyCompare, KeyHash>;
	using key_type = typename cache_impl::key_type;
	using mapped_type = typename cache_impl::mapped_type;

	// @param max_entries_p: A `max_entries` of 0 means that there is no limit on the number of entries in the cache.
	// @param timeout_millisec_p: Timeout in milliseconds for entries, exceeding which invalidates the cache entries; 0
	// means no timeout.
	ThreadSafeWTinyLfuCache(size_t max_entries, uint64_t timeout_millisec)
	    : internal_cache(max_entries, timeout_millisec) {
	}

	// Disable copy and move.
	ThreadSafeWTinyLfuCache(const ThreadSafeWTinyLfuCache &) = delete;
	ThreadSafeWTinyLfuCache &operator=(const ThreadSafeWTinyLfuCache &) = delete;

	~ThreadSafeWTinyLfuCache() = default;

	void Put(Key key, shared_ptr<Val> value) {
		const concurrency::lock_guard<concurrency::mutex> lock(mu);
		internal_cache.Put(std::move(key), std::move(value));
	}

	bool Delete(const Key &key) {
		const concurrency::lock_guard<concurrency::mutex> lock(mu);
		return internal_cache.Delete(key);
	}

	shared_ptr<Val> Get(const Key &key) {
		const concurrency::lock_guard<concurrency::mutex> lock(mu);
		return internal_cache.Get(key);
	}

	void Clear() {
		const concurrency::lock_guard<concurrency::mutex> lock(mu);
		internal_cache.Clear();
	}

	template <typename KeyFilter>
	void Clear(const Key &start_key, KeyFilter &&key_filter) {
		const concurrency::lock_guard<concurrency::mutex> lock(mu);
		internal_cache.Clear(start_key, std::forward<KeyFilter>(key_filter));
	}

	size_t MaxEntries() const {
		return internal_cache.MaxEntries();
	}

	vector<Key> Keys() const {
		const concurrency::lock_guard<concurrency::mutex> lock(mu);
		return internal_cache.Keys();
	}

	vector<pair<Key, shared_ptr<Val>>> Take() {
		const concurrency::lock_guard<concurrency::mutex> lock(mu);
		return internal_cache.Take();
	}

private:
	mutable concurrency::mutex mu;
	WTinyLfuCache<Key, Val, KeyCompare, KeyHash> internal_cache DUCKDB_GUARDED_BY(mu);
};

} // namespace duckdb
