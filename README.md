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

```rust
use hash_rstar::{GeohashRTree, Point, Unique, RstarPoint};

// Define your point type
#[derive(Clone)]
struct Location {
    id: String,
    lat: f64,
    lon: f64,
}

// Implement Point trait for geographic coordinates
impl Point for Location {
    fn point(&self) -> (f64, f64) {
        (self.lon, self.lat)
    }
}

// Implement Unique trait for persistence
impl Unique for Location {
    fn unique_id(&self) -> String {
        self.id.clone()
    }
}

// Implement RstarPoint trait for R-tree operations
impl RstarPoint for Location {
    type Scalar = f64;
    const DIMENSIONS: usize = 2;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Location {
            id: "X".to_string(),
            lon: generator(0),
            lat: generator(1),
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.lon,
            1 => self.lat,
            _ => unreachable!(),
        }
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        match index {
            0 => &mut self.lon,
            1 => &mut self.lat,
            _ => unreachable!(),
        }
    }
}

// Create and use the spatial index
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create new index with geohash precision 6
    let tree = GeohashRTree::new(6, None)?;

    // Insert points
    let point = Location {
        id: "1".into(),
        lon: 116.400357,
        lat: 39.906453,
    };
    tree.insert(point.clone())?;

    // Find nearest neighbor
    if let Some(nearest) = tree.adjacent_cells_nearest(&point)? {
        println!("Found nearest point: {:?}", nearest);
    }
    
    Ok(())
}
```

## Key Technologies

- **Hash Indexing + R* Tree**: Enables efficient Point of Interest (POI) searches by combining spatial and hash-based indexing.
- **`rstar`**: Provides robust spatial indexing for high-performance geospatial queries.
- **`dashmap` (v6.1.0)**: Ensures thread-safe, highly efficient concurrent map operations in Rust.
- **`bincode` (v2.0.1)**: Facilitates fast and compact serialization for data storage and transmission.
- **`sled` (v0.34.7)**: Offers reliable and performant embedded database persistence.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.