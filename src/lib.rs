//! # hash-rstar
//!
//! A concurrent spatial index combining geohash and R-tree data structures for efficient
//! geographic point queries. This library provides thread-safe operations and optional
//! persistence support.
//!
//! ## Features
//!
//! - Concurrent spatial indexing using thread-safe DashMap
//! - Geohash-based spatial partitioning
//! - R-tree for efficient spatial queries within partitions
//! - Optional persistence using sled database
//! - Two nearest neighbor search strategies
//! - Async loading support
//!
//! ## Usage
//!
//! ```rust
//! use hash_rstar::{GeohashRTree, Point, Unique, RstarPoint};
//!
//! // Define your point type
//! #[derive(Clone)]
//! struct Location {
//!     id: String,
//!     lat: f64,
//!     lon: f64,
//! }
//!
//! // Implement Point trait for geographic coordinates
//! impl Point for Location {
//!     fn point(&self) -> (f64, f64) {
//!         (self.lon, self.lat)
//!     }
//! }
//!
//! // Implement Unique trait for persistence
//! impl Unique for Location {
//!     fn unique_id(&self) -> String {
//!         self.id.clone()
//!     }
//! }
//!
//! // Implement RstarPoint trait for R-tree operations
//! impl RstarPoint for Location {
//!     type Scalar = f64;
//!     const DIMENSIONS: usize = 2;
//!
//!     fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
//!         Location {
//!             id: "X".to_string(),
//!             lon: generator(0),
//!             lat: generator(1),
//!         }
//!     }
//!
//!     fn nth(&self, index: usize) -> Self::Scalar {
//!         match index {
//!             0 => self.lon,
//!             1 => self.lat,
//!             _ => unreachable!(),
//!         }
//!     }
//!
//!     fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
//!         match index {
//!             0 => &mut self.lon,
//!             1 => &mut self.lat,
//!             _ => unreachable!(),
//!         }
//!     }
//! }
//!
//! // Create and use the spatial index
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create new index with geohash precision 6
//!     let tree = GeohashRTree::new(6, None)?;
//!
//!     // Insert points
//!     let point = Location {
//!         id: "1".into(),
//!         lon: 116.400357,
//!         lat: 39.906453,
//!     };
//!     tree.insert(point.clone())?;
//!
//!     // Find nearest neighbor
//!     if let Some(nearest) = tree.adjacent_cells_nearest(&point)? {
//!         println!("Found nearest point: {:?}", nearest);
//!     }
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Search Strategies
//!
//! The library provides two nearest neighbor search strategies:
//!
//! 1. `adjacent_cells_nearest`: Fast approximate search
//!    - Searches current and 8 adjacent geohash cells
//!    - Best for evenly distributed points
//!    - Quickest response time
//!
//! 2. `sorted_cells_nearest`: Exact search
//!    - Searches cells in order of increasing distance
//!    - Guarantees finding true nearest neighbor
//!    - Best for unevenly distributed points
//!
//! ## Persistence
//!
//! Optional persistence is supported using sled database:
//!
//! ```rust
//! use std::path::PathBuf;
//!
//! // Create persistent index
//! let tree = GeohashRTree::new(6, Some(PathBuf::from("data.db")))?;
//!
//! // Load existing data
//! let tree = GeohashRTree::load(6, PathBuf::from("data.db"))?;
//!
//! // Async loading
//! let tree = GeohashRTree::load_async(6, PathBuf::from("data.db"))?;
//! ```
//!
//! ## Thread Safety
//!
//! All operations are thread-safe, allowing concurrent access from multiple threads:
//!
//! - Concurrent reads and writes using DashMap
//! - Thread-safe persistence with Arc<sled::Db>
//! - Async loading support
//!
//! ## Performance Considerations
//!
//! - Geohash precision affects partition size and query performance
//! - Higher precision = smaller cells = more precise but slower searches
//! - Lower precision = larger cells = faster but less precise searches
//! - Choose based on data distribution and query patterns
use dashmap::DashMap;
use geo::{Distance, Haversine};
use geohash::{Coord, encode};
use rstar::{Envelope, RTree};
use std::cmp::Ordering;
use std::time::SystemTime;
use std::{path::PathBuf, sync::Arc};
use std::{thread, vec};
use thiserror::Error;

pub use rstar::Point as RstarPoint;
pub use rstar::RTreeObject;

mod utils;

/// A trait for types that can represent a point in geographic coordinates.
///
/// This trait provides functionality for working with geographic points,
/// including methods to get coordinates and calculate distances between points.
///
/// # Examples
///
/// ```
/// use hash_rstar::Point;
///
/// struct Location {
///     lat: f64,
///     lon: f64,
/// }
///
/// impl Point for Location {
///     fn point(&self) -> (f64, f64) {
///         (self.lon, self.lat)
///     }
/// }
/// ```
pub trait Point: RstarPoint {
    /// Returns a tuple containing the x and y coordinates of the point.
    ///
    /// # Returns
    ///
    /// * `(f64, f64)` - A tuple where the first element is the x-coordinate and the second element is the y-coordinate
    fn point(&self) -> (f64, f64);

    /// Calculates the Haversine distance between two points on Earth's surface.
    ///
    /// Returns the distance in meters between this point and another point,
    /// using the Haversine formula which accounts for Earth's spherical shape.
    ///
    /// # Arguments
    ///
    /// * `other` - The other point to calculate distance to
    ///
    /// # Returns
    ///
    /// The distance in meters between the two points
    fn distance(&self, other: &Self) -> f64 {
        let (x, y) = self.point();
        let (other_x, other_y) = other.point();
        Haversine::distance(geo::Point::new(x, y), geo::Point::new(other_x, other_y))
    }

    /// Generates a geohash string representation of the point with specified precision.
    ///
    /// # Arguments
    /// * `geohash_precision` - The length of the geohash string to generate
    ///
    /// # Returns
    /// * `Ok(String)` - The geohash string representation
    /// * `Err(GeohashRTreeError)` - If geohash encoding fails
    fn gen_geohash_str(&self, geohash_precision: usize) -> Result<String, GeohashRTreeError> {
        let (x, y) = self.point();
        let geohash_str = encode(Coord { x, y }, geohash_precision)?;
        Ok(geohash_str)
    }
}

/// A trait for objects that can provide a unique identifier.
///
/// This trait should be implemented by types that need to be uniquely identifiable.
/// The unique identifier is returned as a String value.
pub trait Unique {
    fn unique_id(&self) -> String;
}

/// A concurrent geohash-based R-tree structure for spatial indexing.
///
/// This structure combines geohashing with R-trees to provide efficient spatial queries
/// by partitioning spatial data into geohash cells, each containing an R-tree.
///
/// # Type Parameters
///
/// * `T` - The type of points stored in the tree, must implement required traits for
///         spatial operations, serialization, and thread safety.
///
/// # Fields
///
/// * `arc_dashmap` - Thread-safe concurrent map storing R-trees for each geohash cell
/// * `geohash_precision` - Precision level used for geohash encoding
/// * `db` - Optional persistent storage backend using sled database
pub struct GeohashRTree<T>
where
    T: Unique + Point + Clone + Send + Sync + bincode::Decode<()> + bincode::Encode + 'static,
{
    arc_dashmap: Arc<DashMap<String, RTree<T>>>,
    geohash_precision: usize,
    db: Option<Arc<sled::Db>>,
}

#[derive(Error, Debug)]
pub enum GeohashRTreeError {
    #[error("persistence file not found, {0:?}")]
    LoadError(#[from] sled::Error),
    #[error("geohash error, {0:?}")]
    GeohashError(#[from] geohash::GeohashError),
    #[error("bincode decode error, {0:?}")]
    BincodeDecodeError(#[from] bincode::error::DecodeError),
    #[error("bincode encode error, {0:?}")]
    BincodeEncodeError(#[from] bincode::error::EncodeError),
    #[error("unknown error")]
    Unknown,
}

impl<T> GeohashRTree<T>
where
    T: Unique + Point + Clone + Send + Sync + bincode::Decode<()> + bincode::Encode,
{
    pub fn new(
        geohash_precision: usize,
        persistence_path: Option<PathBuf>,
    ) -> Result<Self, GeohashRTreeError> {
        let mut s = Self {
            arc_dashmap: Arc::new(DashMap::new()),
            geohash_precision,
            db: None,
        };
        if let Some(persistence_path) = persistence_path {
            let db: sled::Db = sled::Config::default().path(persistence_path).open()?;
            s.db = Some(Arc::new(db));
        }
        Ok(s)
    }

    /// Loads a GeohashRTree from a persistence path with the specified geohash precision.
    ///
    /// # Arguments
    ///
    /// * `geohash_precision` - The precision level for geohash calculations
    /// * `persistence_path` - Path to the persistent storage location
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing either the loaded GeohashRTree or a `GeohashRTreeError`
    pub fn load(
        geohash_precision: usize,
        persistence_path: PathBuf,
    ) -> Result<Self, GeohashRTreeError> {
        let db: sled::Db = sled::Config::default().path(persistence_path).open()?;
        let hrt = Self {
            arc_dashmap: Arc::new(DashMap::new()),
            geohash_precision,
            db: Some(Arc::new(db)),
        };

        let config = bincode::config::standard();

        if let Some(db) = &hrt.db {
            let now = SystemTime::now();
            let mut itr = db.iter();
            while let Some(entry) = itr.next() {
                let (_, value) = entry?;
                let (t, _) = bincode::decode_from_slice::<T, _>(&value, config)?;
                let geohash_str = t.gen_geohash_str(geohash_precision)?;
                hrt.arc_dashmap
                    .entry(geohash_str)
                    .or_insert(RTree::new())
                    .insert(t);
            }
            println!(
                "loaded elapsed time: {:?}, total: {}",
                now.elapsed().unwrap(),
                db.len()
            );
        }
        Ok(hrt)
    }

    pub fn load_async(
        geohash_precision: usize,
        persistence_path: PathBuf,
    ) -> Result<Self, GeohashRTreeError> {
        let db: sled::Db = sled::Config::default().path(persistence_path).open()?;
        let hrt = Self {
            arc_dashmap: Arc::new(DashMap::new()),
            geohash_precision,
            db: Some(Arc::new(db)),
        };
        let acr_dashmap = Arc::clone(&hrt.arc_dashmap);

        if let Some(db) = hrt.db.clone() {
            // Load the data from the persistence path
            thread::spawn(move || {
                let config = bincode::config::standard();
                let mut itr = db.iter();
                let now = SystemTime::now();
                while let Some(entry) = itr.next() {
                    let (_, value) = match entry {
                        Ok((uid, value)) => (uid, value),
                        Err(e) => {
                            println!("error: {}", e);
                            return ();
                        }
                    };
                    let t = match bincode::decode_from_slice::<T, _>(&value, config) {
                        Ok((t, _)) => t,
                        Err(e) => {
                            println!("error: {}", e);
                            return ();
                        }
                    };
                    let geohash_str = match t.gen_geohash_str(geohash_precision) {
                        Ok(h) => h,
                        Err(e) => {
                            println!("error: {}", e);
                            return ();
                        }
                    };
                    acr_dashmap
                        .entry(geohash_str)
                        .or_insert(RTree::new())
                        .insert(t);
                }

                println!(
                    "loaded elapsed time: {:?}, total: {}",
                    now.elapsed().unwrap(),
                    db.len()
                );
            });
        }

        Ok(hrt)
    }

    pub fn insert(&self, t: T) -> Result<(), GeohashRTreeError> {
        let geohash_str = t.gen_geohash_str(self.geohash_precision)?;
        if let Some(db) = &self.db {
            let enc = bincode::encode_to_vec(&t, bincode::config::standard())?;
            db.insert(t.unique_id(), enc)?;
        }
        let mut rtree = self.arc_dashmap.entry(geohash_str).or_insert(RTree::new());
        rtree.insert(t);
        Ok(())
    }

    pub fn remove(&self, t: &T) -> Result<Option<T>, GeohashRTreeError> {
        let geohash_str = t.gen_geohash_str(self.geohash_precision)?;
        if let Some(db) = &self.db {
            db.remove(t.unique_id())?;
        }
        let removed = if let Some(mut rtree) = self.arc_dashmap.get_mut(&geohash_str) {
            rtree.remove(t)
        } else {
            None
        };
        Ok(removed)
    }

    /// Finds the nearest neighbor by searching in adjacent geohash cells.
    ///
    /// This method searches for the nearest neighbor in the current geohash cell and its 8 adjacent cells.
    /// It's efficient when the target point is likely to be in neighboring cells, but may miss closer
    /// points that lie just outside the adjacent cells.
    ///
    /// # Algorithm
    /// 1. Searches in the current geohash cell
    /// 2. Searches in all 8 adjacent cells (N, NE, E, SE, S, SW, W, NW)
    /// 3. Returns the closest point among all found points
    ///
    /// # Arguments
    /// * `query_point` - The point to find the nearest neighbor for
    ///
    /// # Returns
    /// * `Ok(Some(T))` - The nearest neighbor if found
    /// * `Ok(None)` - If no neighbor is found
    /// * `Err(GeohashRTreeError)` - If an error occurs during geohash operations
    ///
    /// # Example
    /// ```
    /// let tree = GeohashRTree::new(6, None)?;
    /// let point = Point::new(longitude, latitude);
    /// if let Ok(Some(nearest)) = tree.adjacent_cells_nearest(&point) {
    ///     println!("Found nearest point: {:?}", nearest);
    /// }
    /// ```
    ///
    /// # Use Case
    /// Best suited for scenarios where:
    /// - Points are relatively evenly distributed
    /// - Quick approximate results are acceptable
    /// - The target point is likely close to existing points
    pub fn adjacent_cells_nearest(&self, query_point: &T) -> Result<Option<T>, GeohashRTreeError> {
        let lnglat = query_point.point();
        let geohash_str = encode(
            Coord {
                x: lnglat.0,
                y: lnglat.1,
            },
            self.geohash_precision,
        )?;

        let mut nearests = vec![];
        if let Some(nearest) = self.nearest(query_point, &geohash_str)? {
            nearests.push(nearest);
        }
        let neighbors = geohash::neighbors(&geohash_str)?;
        if let Some(nearest) = self.nearest(query_point, &neighbors.n)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.nearest(query_point, &neighbors.ne)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.nearest(query_point, &neighbors.e)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.nearest(query_point, &neighbors.se)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.nearest(query_point, &neighbors.s)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.nearest(query_point, &neighbors.sw)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.nearest(query_point, &neighbors.w)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.nearest(query_point, &neighbors.nw)? {
            nearests.push(nearest);
        }
        nearests.sort_by(|a, b| {
            let a_distance = a.distance(query_point);
            let b_distance = b.distance(query_point);
            a_distance
                .partial_cmp(&b_distance)
                .unwrap_or(Ordering::Equal)
        });
        Ok(nearests.first().cloned())
    }

    /// Finds the nearest point to the given query point within the specified geohash region.
    ///
    /// # Arguments
    ///
    /// * `query_point` - The reference point to find the nearest neighbor for
    /// * `geohash_str` - The geohash string representing the region to search in
    ///
    /// # Returns
    ///
    /// * `Ok(Some(T))` - The nearest point if found
    /// * `Ok(None)` - If no points exist in the specified geohash region
    /// * `Err(GeohashRTreeError)` - If an error occurs during the search
    fn nearest(&self, query_point: &T, geohash_str: &str) -> Result<Option<T>, GeohashRTreeError> {
        if let Some(rtree) = self.arc_dashmap.get(geohash_str) {
            let nearest = rtree.nearest_neighbor(&query_point.envelope().center());
            return Ok(nearest.cloned());
        }
        Ok(None)
    }

    /// Finds the nearest neighbor using distance-sorted geohash cells.
    ///
    /// This method sorts all possible geohash cells by their minimum distance from the query point
    /// and searches them in order. It guarantees finding the true nearest neighbor but may be slower
    /// than `adjacent_cells_nearest` when searching large areas.
    ///
    /// # Algorithm
    /// 1. Sorts geohash cells by minimum distance from query point
    /// 2. Searches cells in order of increasing distance
    /// 3. Early terminates when finding a point closer than the next cell's minimum distance
    ///
    /// # Arguments
    /// * `query_point` - The point to find the nearest neighbor for
    ///
    /// # Returns
    /// * `Ok(Some(T))` - The nearest neighbor if found
    /// * `Ok(None)` - If no neighbor is found
    /// * `Err(GeohashRTreeError)` - If an error occurs during operations
    ///
    /// # Example
    /// ```
    /// let tree = GeohashRTree::new(5, None)?;
    /// let point = Point::new(longitude, latitude);
    /// if let Ok(Some(nearest)) = tree.sorted_cells_nearest(&point) {
    ///     println!("Found nearest point: {:?}", nearest);
    /// }
    /// ```
    ///
    /// # Use Case
    /// Best suited for scenarios where:
    /// - Points are unevenly distributed
    /// - Exact nearest neighbor is required
    /// - Performance is less critical than accuracy
    /// - Large search areas need to be covered
    pub fn sorted_cells_nearest(&self, query_point: &T) -> Result<Option<T>, GeohashRTreeError> {
        let lnglat = query_point.point();
        let point = geo::point!(x: lnglat.0, y: lnglat.1);
        let sorted_geohash_cells =
            utils::sort_geohash_neighbors(point, self.geohash_precision)?;

        for s in sorted_geohash_cells.iter().enumerate() {
            if let Some(nearest) = self.nearest(query_point, &s.1.0)? {
                let nearest_p = nearest.point();

                // last geohash cell
                if s.0 == sorted_geohash_cells.len() - 1 {
                    return Ok(Some(nearest));
                }

                let dist = Haversine::distance(point, geo::point!(x: nearest_p.0, y: nearest_p.1));
                if dist <= sorted_geohash_cells[s.0 + 1].1 {
                    return Ok(Some(nearest));
                }
            }
        }

        Ok(None)
    }

    pub fn len(&self) -> usize {
        self.arc_dashmap.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::{Decode, Encode};

    #[derive(Clone, PartialEq, Debug, Encode, Decode)]
    struct Location {
        name: String,
        x_coordinate: f64,
        y_coordinate: f64,
    }

    impl Point for Location {
        fn point(&self) -> (f64, f64) {
            (self.x_coordinate, self.y_coordinate)
        }
    }

    impl Unique for Location {
        fn unique_id(&self) -> String {
            self.name.clone()
        }
    }

    impl RstarPoint for Location {
        type Scalar = f64;
        const DIMENSIONS: usize = 2;

        fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
            Location {
                name: "X".to_string(),
                x_coordinate: generator(0),
                y_coordinate: generator(1),
            }
        }

        fn nth(&self, index: usize) -> Self::Scalar {
            match index {
                0 => self.x_coordinate,
                1 => self.y_coordinate,
                _ => unreachable!(),
            }
        }

        fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
            match index {
                0 => &mut self.x_coordinate,
                1 => &mut self.y_coordinate,
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_insert() {
        let hrt: GeohashRTree<Location> = GeohashRTree::new(5, None).unwrap();
        let players = vec![
            Location {
                name: "1".into(),
                x_coordinate: 116.400357,
                y_coordinate: 39.906453,
            },
            Location {
                name: "2".into(),
                x_coordinate: 116.401633,
                y_coordinate: 39.906302,
            },
            Location {
                name: "3".into(),
                x_coordinate: 116.401645,
                y_coordinate: 39.904753,
            },
        ];
        hrt.insert(players[0].clone()).unwrap();
        hrt.insert(players[1].clone()).unwrap();
        hrt.insert(players[2].clone()).unwrap();

        let nearest = hrt
            .adjacent_cells_nearest(&Location {
                name: "1".into(),
                x_coordinate: 116.400357,
                y_coordinate: 39.906453,
            })
            .unwrap()
            .unwrap();

        assert_eq!(players[0], nearest);
    }

    #[test]
    fn test_remove() {
        let hrt: GeohashRTree<Location> = GeohashRTree::new(5, None).unwrap();
        let players = vec![
            Location {
                name: "1".into(),
                x_coordinate: 116.400357,
                y_coordinate: 39.906453,
            },
            Location {
                name: "2".into(),
                x_coordinate: 116.401633,
                y_coordinate: 39.906302,
            },
            Location {
                name: "3".into(),
                x_coordinate: 116.401645,
                y_coordinate: 39.904753,
            },
        ];
        hrt.insert(players[0].clone()).unwrap();
        hrt.insert(players[1].clone()).unwrap();
        hrt.insert(players[2].clone()).unwrap();

        assert_eq!(players[0], hrt.remove(&players[0]).unwrap().unwrap());
        assert_eq!(players[1], hrt.remove(&players[1]).unwrap().unwrap());
        assert_eq!(players[2], hrt.remove(&players[2]).unwrap().unwrap());
    }

    #[test]
    fn test_sorted_cells_nearest() {
        let hrt: GeohashRTree<Location> = GeohashRTree::new(5, None).unwrap();
        let players = vec![
            Location {
                name: "1".into(),
                x_coordinate: 116.400357,
                y_coordinate: 39.906453,
            },
            Location {
                name: "2".into(),
                x_coordinate: 116.401633,
                y_coordinate: 39.906302,
            },
            Location {
                name: "3".into(),
                x_coordinate: 116.401645,
                y_coordinate: 39.904753,
            },
        ];
        hrt.insert(players[0].clone()).unwrap();
        hrt.insert(players[1].clone()).unwrap();
        hrt.insert(players[2].clone()).unwrap();

        let nearest = hrt
            .sorted_cells_nearest(&Location {
                name: "1".into(),
                x_coordinate: 116.400357,
                y_coordinate: 39.906453,
            })
            .unwrap()
            .unwrap();

        assert_eq!(players[0], nearest);
    }
}
