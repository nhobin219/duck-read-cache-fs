#include "catch/catch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>

#include "duckdb/common/string.hpp"
#include "shared_value_lru_cache.hpp"
#include "w_tinylfu_cache.hpp"

using namespace duckdb; // NOLINT

namespace {

// Read [key] through [cache], and insert it on a miss; returns whether it's a cache hit.
template <typename Cache>
bool ReadThrough(Cache &cache, const string &key) {
	if (cache.Get(key) != nullptr) {
		return true;
	}
	cache.Put(key, make_shared_ptr<string>(key));
	return false;
}

// Warm up a hot set, then run a long scan over cold keys with a hot key read after every 10 scanned keys; returns the
// hit ratio of hot key reads during the scan.
template <typename Cache>
double HotKeyHitRatioDuringScan(Cache &cache, size_t hot_key_count, size_t scan_key_count) {
	for (int round = 0; round < 5; ++round) {
		for (size_t idx = 0; idx < hot_key_count; ++idx) {
			ReadThrough(cache, StringUtil::Format("hot-%llu", idx));
		}
	}
	size_t hot_reads = 0;
	size_t hot_hits = 0;
	for (size_t idx = 0; idx < scan_key_count; ++idx) {
		ReadThrough(cache, StringUtil::Format("scan-%llu", idx));
		if (idx % 10 == 0) {
			++hot_reads;
			hot_hits += ReadThrough(cache, StringUtil::Format("hot-%llu", hot_reads % hot_key_count));
		}
	}
	return static_cast<double>(hot_hits) / static_cast<double>(hot_reads);
}

} // namespace

TEST_CASE("WTinyLfu PutAndGetSameKey", "[w-tinylfu test]") {
	ThreadSafeWTinyLfuCache<string, string> cache {/*max_entries=*/1, /*timeout_millisec=*/0};

	// No value initially.
	REQUIRE(cache.Get("1") == nullptr);

	// Check put and get.
	cache.Put("1", make_shared_ptr<string>("1"));
	auto val = cache.Get("1");
	REQUIRE(val != nullptr);
	REQUIRE(*val == "1");

	// Check replacement.
	cache.Put("1", make_shared_ptr<string>("one"));
	val = cache.Get("1");
	REQUIRE(val != nullptr);
	REQUIRE(*val == "one");

	// Check key eviction, the cache holds at most one entry.
	cache.Put("2", make_shared_ptr<string>("2"));
	REQUIRE(cache.Keys().size() == 1);

	// Check deletion.
	const auto keys = cache.Keys();
	REQUIRE(cache.Delete(keys[0]));
	REQUIRE(!cache.Delete(keys[0]));
	REQUIRE(cache.Keys().empty());
}

TEST_CASE("WTinyLfu ScanResistance", "[w-tinylfu test]") {
	constexpr size_t kCapacity = 200;
	constexpr size_t kHotKeys = 100;
	constexpr size_t kScanKeys = 20000;

	// Each hot key is re-read after ~1000 other keys, so an LRU cache of 200 entries has evicted it by then; only the
	// first few hot reads right after warm-up hit.
	SharedValueLruCache<string, string> lru {kCapacity, /*timeout_millisec_p=*/0};
	REQUIRE(HotKeyHitRatioDuringScan(lru, kHotKeys, kScanKeys) < 0.01);

	// W-TinyLFU keeps the hot set, since scanned keys are read only once and lose the admission check.
	WTinyLfuCache<string, string> tinylfu {kCapacity, /*timeout_millisec_p=*/0};
	REQUIRE(HotKeyHitRatioDuringScan(tinylfu, kHotKeys, kScanKeys) > 0.99);
}

TEST_CASE("WTinyLfu AdmitsKeyWhichBecomesFrequent", "[w-tinylfu test]") {
	WTinyLfuCache<string, string> cache {/*max_entries_p=*/100, /*timeout_millisec_p=*/0};
	for (int round = 0; round < 3; ++round) {
		for (int idx = 0; idx < 100; ++idx) {
			ReadThrough(cache, StringUtil::Format("key-%d", idx));
		}
	}

	// Each one-off key pushes "new-key" out of the one-entry window, where it competes with the main region's victim.
	// It's rejected while colder than resident keys, and admitted once its frequency exceeds theirs.
	int misses = 0;
	for (int idx = 0; idx < 20; ++idx) {
		misses += !ReadThrough(cache, "new-key");
		ReadThrough(cache, StringUtil::Format("one-off-%d", idx));
	}
	REQUIRE(misses > 1);
	REQUIRE(misses < 10);
	REQUIRE(cache.Get("new-key") != nullptr);

	// One-off keys never displace resident keys: at most the latest one sits in the window.
	size_t one_off_count = 0;
	for (const auto &key : cache.Keys()) {
		one_off_count += StringUtil::StartsWith(key, "one-off-");
	}
	REQUIRE(one_off_count <= 1);
}

TEST_CASE("WTinyLfu SegmentCapacityInvariant", "[w-tinylfu test]") {
	constexpr size_t kCapacity = 300;
	WTinyLfuCache<string, string> cache {kCapacity, /*timeout_millisec_p=*/0};
	std::mt19937 rng {42};
	// Zipf-like skew: low key ids are much more popular.
	std::uniform_real_distribution<double> dist {0.0, 1.0};
	for (int op = 0; op < 50000; ++op) {
		const auto key_id = static_cast<int>(std::pow(dist(rng), 3.0) * 5000);
		const auto key = std::to_string(key_id);
		if (op % 97 == 0) {
			cache.Delete(key);
		} else {
			ReadThrough(cache, key);
		}
		REQUIRE(cache.WindowSize() + cache.ProbationSize() + cache.ProtectedSize() <= kCapacity);
		REQUIRE(cache.WindowSize() <= 3);
		REQUIRE(cache.ProtectedSize() <= (kCapacity - 3) * 8 / 10);
	}
	REQUIRE(cache.WindowSize() + cache.ProbationSize() + cache.ProtectedSize() == cache.Keys().size());
	REQUIRE(cache.Keys().size() == kCapacity);
}

TEST_CASE("WTinyLfu Timeout", "[w-tinylfu test]") {
	WTinyLfuCache<string, string> cache {/*max_entries_p=*/10, /*timeout_millisec_p=*/50};
	cache.Put("key", make_shared_ptr<string>("val"));
	REQUIRE(cache.Get("key") != nullptr);
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	REQUIRE(cache.Get("key") == nullptr);
	REQUIRE(cache.Keys().empty());
}

TEST_CASE("WTinyLfu Unbounded", "[w-tinylfu test]") {
	WTinyLfuCache<string, string> cache {/*max_entries_p=*/0, /*timeout_millisec_p=*/0};
	for (int idx = 0; idx < 1000; ++idx) {
		cache.Put(std::to_string(idx), make_shared_ptr<string>(std::to_string(idx)));
	}
	REQUIRE(cache.Keys().size() == 1000);
	REQUIRE(*cache.Get("0") == "0");
}

TEST_CASE("WTinyLfu ClearWithFilterAndTake", "[w-tinylfu test]") {
	ThreadSafeWTinyLfuCache<string, string> cache {/*max_entries=*/10, /*timeout_millisec=*/0};
	cache.Put("a-1", make_shared_ptr<string>("a-1"));
	cache.Put("a-2", make_shared_ptr<string>("a-2"));
	cache.Put("b-1", make_shared_ptr<string>("b-1"));
	// Promote one entry into the protected segment, to check deletion works across segments.
	cache.Put("c-1", make_shared_ptr<string>("c-1"));
	REQUIRE(cache.Get("a-1") != nullptr);

	cache.Clear("a-", [](const string &key) { return StringUtil::StartsWith(key, "a-"); });
	auto keys = cache.Keys();
	std::sort(keys.begin(), keys.end());
	REQUIRE(keys == vector<string> {"b-1", "c-1"});

	auto taken = cache.Take();
	REQUIRE(taken.size() == 2);
	REQUIRE(cache.Keys().empty());

	cache.Put("d-1", make_shared_ptr<string>("d-1"));
	cache.Clear();
	REQUIRE(cache.Keys().empty());
}
