use dashmap::DashMap;
use geo::{Distance, Haversine};
use geohash::{Coord, encode};
use rstar::{Envelope, RTree};
use std::cmp::Ordering;
use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};
use std::{thread, vec};
use thiserror::Error;

pub use rstar::Point as RstarPoint;
pub use rstar::RTreeObject;

mod utils;

pub trait Point {
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
        let self_point = self.point();
        let other_point = other.point();
        Haversine::distance(
            geo::Point::new(self_point.0, self_point.1),
            geo::Point::new(other_point.0, other_point.1),
        )
    }
}

pub trait UniqueId: Clone {
    fn unique_id(&self) -> String;
}

pub struct GeohashRTree<T>
where
    T: UniqueId
        + Point
        + RstarPoint
        + Clone
        + Send
        + Sync
        + bincode::Decode<()>
        + bincode::Encode
        + 'static,
{
    arc_dashmap: Arc<DashMap<String, RTree<T>>>,
    geohash_precision: usize,
    db: Option<Arc<sled::Db>>,

    /// A thread-safe boolean flag indicating whether the data structure has been loaded into memory.
    /// Protected by a read-write lock for concurrent access.
    loaded: Arc<RwLock<bool>>,
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
    T: UniqueId + Point + RstarPoint + Clone + Send + Sync + bincode::Decode<()> + bincode::Encode,
{
    pub fn new(geohash_precision: usize, persistence_path: Option<PathBuf>) -> Self {
        let mut s = Self {
            arc_dashmap: Arc::new(DashMap::new()),
            geohash_precision,
            db: None,
            loaded: Arc::new(RwLock::new(true)),
        };
        if let Some(persistence_path) = persistence_path {
            s.db = Some(Arc::new(sled::open(persistence_path).unwrap()));
        }
        s
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
        let config = bincode::config::standard();
        let hrt = Self {
            arc_dashmap: Arc::new(DashMap::new()),
            geohash_precision,
            db: Some(Arc::new(sled::open(persistence_path)?)),
            loaded: Arc::new(RwLock::new(false)),
        };

        if let Some(db) = &hrt.db {
            // Load the data from the persistence path
            let mut itr = db.iter();
            while let Some(entry) = itr.next() {
                let (geohash_key, value) = entry?;
                let vals: Vec<T> = bincode::decode_from_slice(&value, config)?.0;
                hrt.arc_dashmap.insert(
                    geohash_key.escape_ascii().to_string(),
                    RTree::bulk_load(vals.clone()),
                );
            }
        }

        *hrt.loaded.write().unwrap() = true;
        Ok(hrt)
    }

    pub fn load_async(
        geohash_precision: usize,
        persistence_path: PathBuf,
    ) -> Result<Self, GeohashRTreeError> {
        let hrt = Self {
            arc_dashmap: Arc::new(DashMap::new()),
            geohash_precision,
            db: Some(Arc::new(sled::open(persistence_path)?)),
            loaded: Arc::new(RwLock::new(false)),
        };

        let acr_dashmap = Arc::clone(&hrt.arc_dashmap);
        let arc_loaded = Arc::clone(&hrt.loaded);

        if let Some(db) = hrt.db.clone() {
            // Load the data from the persistence path
            thread::spawn(move || {
                let config = bincode::config::standard();
                let mut itr = db.iter();
                while let Some(entry) = itr.next() {
                    let (geohash_key, value) = match entry {
                        Ok((geohash_key, value)) => (geohash_key, value),
                        Err(e) => {
                            println!("error: {}", e);
                            return ();
                        }
                    };

                    match bincode::decode_from_slice(&value, config) {
                        Ok((vals, _)) => {
                            acr_dashmap.insert(
                                geohash_key.escape_ascii().to_string(),
                                RTree::bulk_load(vals),
                            );
                        }
                        Err(e) => {
                            println!("error: {}", e);
                        }
                    }
                }

                *arc_loaded.write().unwrap() = true;
                println!("loaded, len: {}", acr_dashmap.len());
            });
        }

        Ok(hrt)
    }

    pub fn insert(&self, t: T) -> Result<(), GeohashRTreeError> {
        let lnglat = t.point();
        let geohash_str = encode(
            Coord {
                x: lnglat.0,
                y: lnglat.1,
            },
            self.geohash_precision,
        )?;

        // key founded
        if let Some(mut rtree) = self.arc_dashmap.get_mut(&geohash_str) {
            rtree.insert(t);
        } else if *self.loaded.read().unwrap() {
            // loaded but key not found
            self.arc_dashmap
                .insert(geohash_str.clone(), RTree::bulk_load(vec![t]));
        } else {
            // load from persistence db
            if let Some(db) = &self.db {
                if let Some(data) = db.get(&geohash_str)? {
                    let dec: (Vec<T>, _) =
                        bincode::decode_from_slice(&data, bincode::config::standard())?;
                    let mut rtree = RTree::bulk_load(dec.0);
                    rtree.insert(t);
                    self.arc_dashmap.insert(geohash_str.clone(), rtree);
                }
            }
        }

        if let Some(db) = &self.db {
            if let Some(rtree) = self.arc_dashmap.get_mut(&geohash_str) {
                let mut vals: Vec<T> = Vec::with_capacity(rtree.size());
                rtree.iter().for_each(|v| vals.push(v.clone()));
                let enc = bincode::encode_to_vec(vals, bincode::config::standard())?;
                db.insert(&geohash_str, enc)?;
            }
        }

        Ok(())
    }

    pub fn remove(&self, t: T) -> Result<(), GeohashRTreeError> {
        let lnglat = t.point();
        let geohash_str = encode(
            Coord {
                x: lnglat.0,
                y: lnglat.1,
            },
            self.geohash_precision,
        )?;

        // key founded
        if let Some(mut rtree) = self.arc_dashmap.get_mut(&geohash_str) {
            rtree.remove(&t);
        } else if !*self.loaded.read().unwrap() {
            if let Some(db) = &self.db {
                if let Some(data) = db.get(&geohash_str)? {
                    let dec: (Vec<T>, _) =
                        bincode::decode_from_slice(&data, bincode::config::standard())?;
                    let mut rtree = RTree::bulk_load(dec.0);
                    rtree.remove(&t);
                    self.arc_dashmap.insert(geohash_str.clone(), rtree);
                }
            }
        }

        if let Some(db) = &self.db {
            if let Some(rtree) = self.arc_dashmap.get_mut(&geohash_str) {
                let mut vals: Vec<T> = Vec::with_capacity(rtree.size());
                rtree.iter().for_each(|v| vals.push(v.clone()));
                let enc = bincode::encode_to_vec(vals, bincode::config::standard())?;
                db.insert(&geohash_str, enc)?;
            }
        }
        Ok(())
    }

    /// Finds the nearest neighbor to the given query point by searching in the current geohash cell
    /// and its adjacent cells.
    ///
    /// # Arguments
    ///
    /// * `query_point` - The point to find the nearest neighbor for
    ///
    /// # Returns
    ///
    /// * `Ok(Some(T))` - The nearest neighbor if found
    /// * `Ok(None)` - If no neighbor is found
    /// * `Err(GeohashRTreeError)` - If an error occurs during geohash operations
    ///
    /// # Example
    ///
    /// ```
    /// let tree = GeohashRTree::new(6);
    /// let point = Point::new(longitude, latitude);
    /// if let Ok(Some(nearest)) = tree.nearest_neighbor(&point) {
    ///     println!("Found nearest point: {:?}", nearest);
    /// }
    /// ```
    pub fn nearest_neighbor(&self, query_point: &T) -> Result<Option<T>, GeohashRTreeError> {
        let lnglat = query_point.point();
        let geohash_str = encode(
            Coord {
                x: lnglat.0,
                y: lnglat.1,
            },
            self.geohash_precision,
        )?;

        let mut nearests = vec![];
        if let Some(nearest) = self.get_nearby_geohash(query_point, &geohash_str)? {
            nearests.push(nearest);
        }
        let neighbors = geohash::neighbors(&geohash_str)?;
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.n)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.ne)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.e)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.se)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.s)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.sw)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.w)? {
            nearests.push(nearest);
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.nw)? {
            nearests.push(nearest);
        }
        nearests.sort_by(|a, b| {
            let a_distance = a.distance(query_point);
            let b_distance = b.distance(query_point);
            a_distance
                .partial_cmp(&b_distance)
                .unwrap_or(Ordering::Equal)
        });
        Ok(None)
    }

    /// Retrieves the nearest point to the query point within a specific geohash cell.
    ///
    /// If the point is not found in the in-memory cache, it will attempt to load the data from the database.
    /// A data is loaded to a rtree.
    ///
    ///
    /// # Arguments
    ///
    /// * `query_point` - The reference point to find the nearest neighbor
    /// * `geohash_str` - The geohash string representing the cell to search in
    ///
    /// # Returns
    ///
    /// * `Result<Option<T>, GeohashRTreeError>` - Returns the nearest point if found, None if no points exist in the cell,
    ///   or an error if database operations fail
    ///
    /// # Note
    ///
    /// First checks in-memory cache (arc_dashmap), then falls back to database if configured.
    /// Loads data from database into memory cache when accessed for the first time.
    fn get_nearby_geohash(
        &self,
        query_point: &T,
        geohash_str: &str,
    ) -> Result<Option<T>, GeohashRTreeError> {
        if let Some(rtree) = self.arc_dashmap.get(geohash_str) {
            let nearest = rtree.nearest_neighbor(&query_point.envelope().center());
            return Ok(nearest.cloned());
        }
        if *self.loaded.read().unwrap() {
            return Ok(None);
        }
        if let Some(db) = &self.db {
            if let Some(data) = db.get(geohash_str)? {
                let config = bincode::config::standard();
                let dec: (Vec<T>, _) = bincode::decode_from_slice(&data, config)?;
                let rtree = RTree::bulk_load(dec.0);
                let nearest = rtree.nearest_neighbor(&query_point.envelope().center());
                let nearest = nearest.cloned();
                self.arc_dashmap.insert(geohash_str.to_string(), rtree);
                return Ok(nearest);
            }
        }
        Ok(None)
    }

    /// Finds the nearest neighbor to the given query point using geohash-based search.
    ///
    /// # Arguments
    ///
    /// * `query_point` - The point to find the nearest neighbor for, implementing the required trait.
    ///
    /// # Returns
    ///
    /// * `Result<Option<T>, GeohashRTreeError>` - Returns the nearest neighbor if found, None if no neighbors exist,
    ///   or an error if the operation fails.
    ///
    /// # Example
    ///
    /// ```
    /// let tree = GeohashRTree::new(5);
    /// let point = Point::new(longitude, latitude);
    /// let nearest = tree.nearest_neighbor_2(&point)?;
    /// ```
    pub fn nearest_neighbor_2(&self, query_point: &T) -> Result<Option<T>, GeohashRTreeError> {
        let lnglat = query_point.point();
        let point = geo::point!(x: lnglat.0, y: lnglat.1);
        let sorted_geohash_cells =
            utils::sort_geohash_cells_by_min_distance(point, self.geohash_precision)?;

        for s in sorted_geohash_cells.iter().enumerate() {
            if let Some(nearest) = self.get_nearby_geohash(query_point, &s.1.0)? {
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

    pub fn nearest_neighbor_iter_with_distance_2(
        &self,
        query_point: &T,
    ) -> Result<Vec<T>, GeohashRTreeError> {
        let lnglat = query_point.point();
        let geohash_str = encode(
            Coord {
                x: lnglat.0,
                y: lnglat.1,
            },
            self.geohash_precision,
        )?;
        if let Some(rtree) = self.arc_dashmap.get(&geohash_str) {
            let nearest_iter =
                rtree.nearest_neighbor_iter_with_distance_2(&query_point.envelope().center());
            return Ok(nearest_iter.map(|iter| iter.0.clone()).collect());
        }
        Ok(vec![])
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
    struct Player {
        name: String,
        x_coordinate: f64,
        y_coordinate: f64,
    }

    impl Point for Player {
        fn point(&self) -> (f64, f64) {
            (self.x_coordinate, self.y_coordinate)
        }
    }

    impl UniqueId for Player {
        fn unique_id(&self) -> String {
            self.name.clone()
        }
    }

    impl RstarPoint for Player {
        type Scalar = f64;
        const DIMENSIONS: usize = 2;

        fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
            Player {
                name: "Player".to_string(),
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
    fn test_geohash_rtree() {
        let hrt: GeohashRTree<Player> = GeohashRTree::new(5, None);
        let players = vec![
            Player {
                name: "1".into(),
                x_coordinate: 116.400357,
                y_coordinate: 39.906453,
            },
            Player {
                name: "2".into(),
                x_coordinate: 116.401633,
                y_coordinate: 39.906302,
            },
            Player {
                name: "3".into(),
                x_coordinate: 116.401645,
                y_coordinate: 39.904753,
            },
        ];
        hrt.insert(players[0].clone()).unwrap();
        hrt.insert(players[1].clone()).unwrap();
        hrt.insert(players[2].clone()).unwrap();

        let nearest = hrt
            .nearest_neighbor(&Player {
                name: "1".into(),
                x_coordinate: 116.400357,
                y_coordinate: 39.906453,
            })
            .unwrap()
            .unwrap();

        println!("nearest: {:?}", nearest);
        assert_eq!(players[0], nearest)
    }
}
