// In-memory cache cache block key.

#pragma once

#include <functional>
#include <tuple>

#include "duckdb/common/string.hpp"
#include "duckdb/common/typedefs.hpp"

namespace duckdb {

struct InMemCacheBlock {
	// Remote filepath.
	string fname;
	idx_t start_off = 0;
	idx_t blk_size = 0;

	InMemCacheBlock(const string &path, idx_t start_off_p, idx_t blk_size_p);
};

struct InMemCacheBlockLess {
	bool operator()(const InMemCacheBlock &lhs, const InMemCacheBlock &rhs) const {
		return std::tie(lhs.fname, lhs.start_off, lhs.blk_size) < std::tie(rhs.fname, rhs.start_off, rhs.blk_size);
	}
};
struct InMemCacheBlockEqual {
	bool operator()(const InMemCacheBlock &lhs, const InMemCacheBlock &rhs) const {
		return std::tie(lhs.fname, lhs.start_off, lhs.blk_size) == std::tie(rhs.fname, rhs.start_off, rhs.blk_size);
	}
};
struct InMemCacheBlockHash {
	std::size_t operator()(const InMemCacheBlock &key) const {
		std::size_t hash = std::hash<string> {}(key.fname);
		hash ^= std::hash<idx_t> {}(key.start_off) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
		hash ^= std::hash<idx_t> {}(key.blk_size) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
		return hash;
	}
};

} // namespace duckdb
