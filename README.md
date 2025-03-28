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

## Usage

Below is an example of how to use `hash-rstar`:

```rust
use hash_rstar::{GeohashRTree, GeohashRTreeObject};

#[derive(Debug, PartialEq, Clone, bincode::Encode, bincode::Decode)]
struct Location {
    id: String,
    x_coordinate: f64,
    y_coordinate: f64,
}

impl GeohashRTreeObject for Location {
    fn unique_id(&self) -> String {
        self.id.clone()
    }

    fn x_y(&self) -> (f64, f64) {
        (self.x_coordinate, self.y_coordinate)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new GeohashRTree with precision 5 and no persistence
    let tree: GeohashRTree<Location> = GeohashRTree::new(5, None)?;

    // Insert a location into the tree
    let location = Location {
        id: "1".into(),
        x_coordinate: 116.400357,
        y_coordinate: 39.906453,
    };
    tree.insert(location.clone())?;

    // Find the nearest neighbor
    if let Some(nearest) = tree.adjacent_cells_nearest(&location)? {
        println!("Found nearest point: {:?}", nearest);
    }

    Ok(())
}
```

## Key Technologies

- **Hash Indexing + R-Tree**: Enables efficient Point of Interest (POI) searches by combining spatial and hash-based indexing.
- **`rstar`**: Provides robust spatial indexing for high-performance geospatial queries.
- **`dashmap`**: Ensures thread-safe, highly efficient concurrent map operations in Rust.
- **`bincode`**: Facilitates fast and compact serialization for data storage and transmission.
- **`sled`**: Offers reliable and performant embedded database persistence.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.