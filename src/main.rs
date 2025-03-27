//! # Hash-RTree Spatial Service
//!
//! A REST API service that provides spatial querying capabilities using a combination of
//! geohash-based indexing and R-tree data structures.
//!
//! ## Features
//!
//! * Nearest neighbor queries for geographic locations
//! * Add multiple locations in a single request
//! * Delete locations by ID
//! * Persistent storage with configurable location
//! * CORS support for web applications
//!
//! ## API Endpoints
//!
//! * `GET /nearest/{lon,lat}` - Find nearest location to given coordinates
//! * `POST /locations` - Add multiple locations
//! * `DELETE /locations/{id1,id2,...}` - Delete locations by IDs
//!
//! ## Configuration
//!
//! Environment variables:
//! * `HASH_RSTAR_DB` - Database file path
//! * `HASH_RSTAR_GEOHASH_PRECISION` - Geohash precision (default: 5)
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::{delete, get, post},
};

use bincode::{Decode, Encode};
use hash_rstar::{GeohashRTree, Point, RstarPoint, Unique};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::{collections::HashMap, sync::Arc};
use tower_http::cors::{Any, CorsLayer};

#[derive(Clone, PartialEq, Debug, Encode, Decode, Deserialize, Serialize, Default)]
struct Location {
    id: String,
    name: String,
    lon_lat: (f32, f32),
    extra: Option<HashMap<String, String>>,
}

impl Point for Location {
    fn point(&self) -> (f32, f32) {
        self.lon_lat
    }
}

impl Unique for Location {
    fn unique_id(&self) -> String {
        self.id.clone()
    }
}

impl RstarPoint for Location {
    type Scalar = f32;
    const DIMENSIONS: usize = 2;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Location {
            id: "".to_string(),
            name: "".to_string(),
            lon_lat: (generator(0), generator(1)),
            extra: None,
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.lon_lat.0,
            1 => self.lon_lat.1,
            _ => unreachable!(),
        }
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        match index {
            0 => &mut self.lon_lat.0,
            1 => &mut self.lon_lat.1,
            _ => unreachable!(),
        }
    }
}

struct AppState {
    hash_rtree: GeohashRTree<Location>,
}

#[tokio::main]
async fn main() {
    let default_db_path = if let Ok(db) = std::env::var("HASH_RSTAR_DB") {
        db
    } else if cfg!(windows) {
        let program_data = std::env::var("PROGRAMDATA").unwrap_or("C:\\ProgramData".to_string());
        format!("{}\\hash-rstar\\data", program_data)
    } else {
        "/var/lib/hash-rstar/data".to_string()
    };

    let geohash_precision: usize = std::env::var("HASH_RSTAR_GEOHASH_PRECISION")
        .unwrap_or("5".to_string())
        .parse()
        .unwrap();

    let shared_state = Arc::new(AppState {
        hash_rtree: GeohashRTree::load(geohash_precision, PathBuf::from(&default_db_path)).unwrap(),
    });

    let app = Router::new()
        .route("/nearest/{lonlat}", get(nearest))
        .route("/locations/{lonlat}", delete(delete_locations))
        .route("/locations", post(add_locations))
        .with_state(Arc::clone(&shared_state))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn parse_lonlat(lonlat: &str) -> Result<(f32, f32), String> {
    let mut parts = lonlat.split(',');
    match (parts.next(), parts.next()) {
        (Some(lon), Some(lat)) => {
            let lon = lon
                .parse::<f32>()
                .map_err(|e| format!("Invalid longitude: {}, {}", lon, e.to_string()))?;
            let lat = lat
                .parse::<f32>()
                .map_err(|e| format!("Invalid latitude: {}, {}", lat, e.to_string()))?;
            if (-180.0..=180.0).contains(&lon) && (-90.0..=90.0).contains(&lat) {
                Ok((lon, lat))
            } else {
                Err(format!(
                    "Coordinates out of range: longitude {}, latitude {}",
                    lon, lat
                ))
            }
        }
        _ => Err(format!("Invalid lonlat format: {}", lonlat)),
    }
}

async fn nearest(
    State(state): State<Arc<AppState>>,
    Path(lonlat): Path<String>,
) -> Result<Json<Option<Location>>, (StatusCode, String)> {
    let mut query = Location::default();
    query.lon_lat = parse_lonlat(&lonlat).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let n = state
        .hash_rtree
        .sorted_cells_nearest(&query)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(n))
}

async fn delete_locations(
    State(state): State<Arc<AppState>>,
    Path(ids): Path<String>,
) -> Result<Json<Vec<Option<Location>>>, (StatusCode, String)> {
    let mut deleted = vec![];
    for id in ids.split(",") {
        let d = state
            .hash_rtree
            .delete(id)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        deleted.push(d);
    }
    Ok(Json(deleted))
}

async fn add_locations(
    State(state): State<Arc<AppState>>,
    Json(locations): Json<Vec<Location>>,
) -> Result<(), (StatusCode, String)> {
    for lct in locations {
        state
            .hash_rtree
            .insert(lct)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    Ok(())
}
