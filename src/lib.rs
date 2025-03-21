use dashmap::DashMap;
use geohash::{Coord, encode};
use rstar::{Envelope, RTree};
use std::{path::PathBuf, sync::Arc};
use thiserror::Error;

pub use rstar::Point as RstarPoint;
pub use rstar::RTreeObject;

pub trait Point {
    fn point(&self) -> (f64, f64);
}

pub trait UniqueId: Clone {
    fn unique_id(&self) -> String;
}

pub struct GeohashRTree<T>
where
    T: UniqueId + Point + RstarPoint + Clone + Send + Sync + bincode::Decode<()> + bincode::Encode,
{
    arc_dashmap: Arc<DashMap<String, RTree<T>>>,
    geohash_precision: usize,
    db: Option<sled::Db>,
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
        };
        if let Some(persistence_path) = persistence_path {
            s.db = Some(sled::open(persistence_path).unwrap());
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
        let hrt = Self::new(geohash_precision, Some(persistence_path));

        if let Some(db) = &hrt.db {
            // Load the data from the persistence path
            let mut itr = db.iter();
            while let Some(entry) = itr.next() {
                let (geohash_key, value) = entry?;
                let vals: Vec<T> = bincode::decode_from_slice(&value, config)?.0;
                hrt.arc_dashmap.insert(
                    geohash_key.escape_ascii().to_string(),
                    RTree::bulk_load(vals),
                );
            }
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

        let mut rtree = self
            .arc_dashmap
            .entry(geohash_str.clone())
            .or_insert_with(RTree::new);
        rtree.insert(t);

        if let Some(db) = &self.db {
            let config = bincode::config::standard();
            let mut vals: Vec<T> = Vec::with_capacity(rtree.size());
            rtree.iter().for_each(|v| vals.push(v.clone()));
            let enc = bincode::encode_to_vec(vals, config)?;
            db.insert(&geohash_str, enc)?;
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
        if let Some(mut rtree) = self.arc_dashmap.get_mut(&geohash_str) {
            if let Some(_) = rtree.remove(&t) {
                if let Some(db) = &self.db {
                    let config = bincode::config::standard();
                    let mut vals: Vec<T> = Vec::with_capacity(rtree.size());
                    rtree.iter().for_each(|v| vals.push(v.clone()));
                    let enc = bincode::encode_to_vec(vals, config)?;
                    db.insert(&geohash_str, enc)?;
                }
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

        if let Some(nearest) = self.get_nearby_geohash(query_point, &geohash_str)? {
            return Ok(Some(nearest.clone()));
        }
        let neighbors = geohash::neighbors(&geohash_str)?;
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.n)? {
            return Ok(Some(nearest.clone()));
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.ne)? {
            return Ok(Some(nearest.clone()));
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.e)? {
            return Ok(Some(nearest.clone()));
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.se)? {
            return Ok(Some(nearest.clone()));
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.s)? {
            return Ok(Some(nearest.clone()));
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.sw)? {
            return Ok(Some(nearest.clone()));
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.w)? {
            return Ok(Some(nearest.clone()));
        }
        if let Some(nearest) = self.get_nearby_geohash(query_point, &neighbors.nw)? {
            return Ok(Some(nearest.clone()));
        }
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
