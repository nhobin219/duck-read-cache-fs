#include "extension_bounded_data_cache_storage.hpp"

#include "duckdb/common/helper.hpp"

namespace duckdb {

template <typename Cache>
ExtensionBoundedDataCacheStorage<Cache>::ExtensionBoundedDataCacheStorage(size_t max_entries, uint64_t timeout_millisec)
    : cache(max_entries, timeout_millisec) {
}

template <typename Cache>
void ExtensionBoundedDataCacheStorage<Cache>::Put(InMemCacheBlock key, PageAlignedDataChunk chunk, string version_tag) {
	auto entry = make_shared_ptr<InMemCacheDataEntry>();
	entry->data = std::move(chunk);
	entry->version_tag = std::move(version_tag);
	cache.Put(std::move(key), std::move(entry));
}

template <typename Cache>
optional<PinnedBlock> ExtensionBoundedDataCacheStorage<Cache>::Get(const InMemCacheBlock &key,
                                                                   const string &expected_version_tag) {
	auto entry = cache.Get(key);
	if (entry == nullptr) {
		return nullopt;
	}
	if (!PinnedBlock::ValidateVersionTag(entry->version_tag, expected_version_tag)) {
		cache.Delete(key);
		return nullopt;
	}

	const PageAlignedDataChunk *chunk_ptr = &entry->data;
	shared_ptr<void> keep_alive = std::move(entry);
	return PinnedBlock {std::move(keep_alive), chunk_ptr};
}

template <typename Cache>
bool ExtensionBoundedDataCacheStorage<Cache>::Delete(const InMemCacheBlock &key) {
	return cache.Delete(key);
}

template <typename Cache>
void ExtensionBoundedDataCacheStorage<Cache>::Clear() {
	cache.Clear();
}

template <typename Cache>
void ExtensionBoundedDataCacheStorage<Cache>::Clear(const InMemCacheBlock &start_key,
                                                    std::function<bool(const InMemCacheBlock &)> filter) {
	cache.Clear(start_key, std::move(filter));
}

template <typename Cache>
vector<InMemCacheBlock> ExtensionBoundedDataCacheStorage<Cache>::Keys() const {
	return cache.Keys();
}

template <typename Cache>
vector<std::pair<InMemCacheBlock, shared_ptr<InMemCacheDataEntry>>> ExtensionBoundedDataCacheStorage<Cache>::Take() {
	return cache.Take();
}

template class ExtensionBoundedDataCacheStorage<
    ThreadSafeSharedValueLruCache<InMemCacheBlock, InMemCacheDataEntry, InMemCacheBlockLess>>;
template class ExtensionBoundedDataCacheStorage<
    ThreadSafeWTinyLfuCache<InMemCacheBlock, InMemCacheDataEntry, InMemCacheBlockLess, InMemCacheBlockHash>>;

} // namespace duckdb
