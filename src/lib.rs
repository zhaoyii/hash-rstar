use dashmap::DashMap;
use geohash::{Coord, encode};
use rstar::RTree;
use std::{hash::Hash, path::PathBuf, sync::Arc};
use thiserror::Error;

pub use rstar::Point as RstarPoint;

pub trait Point {
    fn point(&self) -> (f64, f64);
}

pub struct GeohashRTree<K, T>
where
    K: AsRef<[u8]> + Eq + Hash + From<String>,
    T: Point + rstar::Point + Clone + Send + Sync + bincode::Decode<()>,
{
    arc_dashmap: Arc<DashMap<K, RTree<T>>>,
    geohash_precision: usize,
    persistence_path: Option<PathBuf>,
}

#[derive(Error, Debug)]
pub enum GeohashRTreeError {
    #[error("persistence file not found, {0:?}")]
    LoadError(#[from] sled::Error),
    #[error("geohash error, {0:?}")]
    GeohashError(#[from] geohash::GeohashError),
    #[error("unknown error")]
    Unknown,
}

impl<K, T> GeohashRTree<K, T>
where
    K: AsRef<[u8]> + Eq + Hash + From<String>,
    T: Point + rstar::Point + Clone + Send + Sync + bincode::Decode<()>,
{
    /// Creates a new instance of the struct with the specified geohash precision
    /// and an optional persistence path.
    ///
    /// # Parameters
    /// - `geohash_precision`: The precision level for geohashing, represented as an 8-bit unsigned integer.
    /// - `persistence_path`: An optional `PathBuf` specifying the path for persistence storage. If `None`,
    ///   persistence is disabled.
    ///
    /// # Returns
    /// A new instance of the struct initialized with the provided geohash precision and persistence path.
    ///
    /// # Examples
    /// ```
    /// use std::path::PathBuf;
    ///
    /// let instance = GeohashRTree::new(8, Some(PathBuf::from("path/to/persistence")));
    /// ```
    pub fn new(geohash_precision: usize, persistence_path: Option<PathBuf>) -> Self {
        Self {
            arc_dashmap: Arc::new(DashMap::new()),
            geohash_precision,
            persistence_path,
        }
    }

    /// 加载持久化数据的方法，
    /// 如果持久化路径存在，则从持久化路径加载数据，
    /// 否则返回错误。
    pub fn load(&self) -> Result<usize, GeohashRTreeError> {
        let mut count: usize = 0;
        let config = bincode::config::standard();

        if let Some(path) = &self.persistence_path {
            // Load the data from the persistence path
            let db: sled::Db = sled::open(path)?;

            let mut itr = db.iter();
            while let Some(entry) = itr.next() {
                let (_, value) = entry?;
                let t: T = bincode::decode_from_slice(&value, config)
                    .map_err(|_| GeohashRTreeError::Unknown)?
                    .0;
                let lnglat = t.point();
                let geohash = encode(
                    Coord {
                        x: lnglat.0,
                        y: lnglat.1,
                    },
                    self.geohash_precision,
                )?;

                let mut rtree = self
                    .arc_dashmap
                    .entry(geohash.into())
                    .or_insert_with(RTree::new);
                rtree.insert(t);
            }
            count = db.len();
        }

        Ok(count)
    }

    pub fn insert(&self, t: T) -> Result<(), GeohashRTreeError> {
        let lnglat = t.point();
        let geohash = encode(
            Coord {
                x: lnglat.0,
                y: lnglat.1,
            },
            self.geohash_precision,
        )?;

        let mut rtree = self
            .arc_dashmap
            .entry(geohash.into())
            .or_insert_with(RTree::new);
        rtree.insert(t);

        Ok(())
    }

    pub fn nearest_neighbor(&self, query_point: &T) -> Result<Option<T>, GeohashRTreeError> {
        let lnglat = query_point.point();
        let geohash = encode(
            Coord {
                x: lnglat.0,
                y: lnglat.1,
            },
            self.geohash_precision,
        )?;
        if let Some(rtree) = self.arc_dashmap.get(&geohash.into()) {
            let nearest = rtree.nearest_neighbor(query_point);
            return Ok(nearest.cloned());
        }
        Ok(None)
    }

    pub fn nearest_neighbor_iter_with_distance_2(
        &self,
        query_point: &T,
    ) -> Result<Vec<T>, GeohashRTreeError> {
        let lnglat = query_point.point();
        let geohash = encode(
            Coord {
                x: lnglat.0,
                y: lnglat.1,
            },
            self.geohash_precision,
        )?;
        if let Some(rtree) = self.arc_dashmap.get(&geohash.into()) {
            let nearest_iter = rtree.nearest_neighbor_iter_with_distance_2(query_point);
            return Ok(nearest_iter.map(|iter| iter.0.clone()).collect());
        }
        Ok(vec![])
    }
}
