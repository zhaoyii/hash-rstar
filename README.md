## hash-rstar

### Overview

`hash-rstar` is a high-performance hash index database built on top of the R* tree. It is designed for developers seeking efficient solutions for reverse geocoding.

### Background
Current reverse geocoding databases face several challenges:

- [Redis GEOSEARCH](https://redis.io/docs/latest/commands/geosearch/) struggles with performance over long distances, even when using hash-based indexing.
- [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/geo-queries.html) relies on disk storage, which can result in slower query performance.
- [Tile38](https://github.com/tidwall/tile38), an open-source (MIT licensed) geolocation data store and spatial index, exhibits slower query speeds without hash-based indexing.

By combining hash indexing with the R* tree, a Rust-based implementation achieves outstanding in-memory computation performance, addressing these limitations effectively.

### Features

- **High Query Speed**: Combines hash indexing with the R* tree for exceptional query performance.
- **Persistence with Lazy Loading**: Supports data persistence and efficient lazy loading mechanisms.
- **Thread-Safe**: Ensures thread-safe, highly efficient concurrent map operations.

### Installation

### Usage

### Contributing

## Key Technologies

- **Hash Indexing + R* Tree**: Enables efficient Point of Interest (POI) searches by combining spatial and hash-based indexing.
- **`rstar`**: Provides robust spatial indexing for high-performance geospatial queries.
- **`dashmap` (v6.1.0)**: Ensures thread-safe, highly efficient concurrent map operations in Rust.
- **`bincode` (v2.0.1)**: Facilitates fast and compact serialization for data storage and transmission.
- **`sled` (v0.34.7)**: Offers reliable and performant embedded database persistence.

### License
