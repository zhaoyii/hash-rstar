# Hash-RTree Spatial Service

A REST API service that provides spatial querying capabilities using a combination of geohash-based indexing and R-tree data structures.

## Features

- Nearest neighbor queries for geographic locations
- Add multiple locations in a single request
- Delete locations by ID
- Persistent storage with configurable location
- CORS support for web applications

## Getting Started

### Pull the Image

To pull the latest version of the image from Docker Hub:

```bash
docker pull zhaoyii/hash-rstar:latest
```

### Run the Container

Run the container with the following command:

```bash
docker run -d \
  -p 3000:3000 \
  -e HASH_RSTAR_DB=/data/hash_rstar.db \
  -e HASH_RSTAR_GEOHASH_PRECISION=5 \
  -v /path/to/local/data:/var/lib/hash-rstar/data \
  zhaoyii/hash-rstar:latest
```

### Environment Variables
- `HASH_RSTAR_DB`: Path to the database file (default: `/var/lib/hash-rstar/data`).
- `HASH_RSTAR_GEOHASH_PRECISION`: Geohash precision for indexing (default: `5`).

### Volume Mounts
`/var/lib/hash-rstar/data`: Mount a local directory to persist the database file.

### API Endpoints

1. Find Nearest Location: `GET /nearest/{lon,lat}`, Example:
```bash
curl -X GET "http://localhost:3000/nearest/-73.935242,40.730610"
```

2. Add Locations: `POST /locations`, Example:
```bash
curl -X POST \
  http://localhost:3000/locations \
  -H "Content-Type: application/json" \
  -d '[
    {
      "id": "1",
      "name": "Location1",
      "lon_lat": [-73.935242, 40.730610],
      "extra": {"type": "park", "rating": 4.5}
    },
    {
      "id": "2",
      "name": "Location2",
      "lon_lat": [-73.955242, 40.750610],
      "extra": {"type": "restaurant", "rating": 4.0}
    }
  ]' 
```

**Field Descriptions:**
- `id` (string): Unique identifier for the location.
- `name` (string): Name of the location.
- `lon_lat` (array): Longitude and latitude of the location in `[longitude, latitude]` format.
- `extra` (object, optional): A map containing additional metadata about the location. Example keys include:
  - `type` (string): Type of the location (e.g., "park", "restaurant").
  - `rating` (number): Rating of the location.


3. Delete Locations: `DELETE /locations/{id1,id2,...}`, Example:
```bash
curl -X DELETE "http://localhost:3000/locations/1,2"
```

## Build Locally (Optional)

If you want to build the Docker image locally:

1. Clone the repository:
```bash
git clone https://github.com/zhaoyii/hash-rstar.git
cd hash-rstar
```
2. Build the Docker image:
```bash
docker build -t zhaoyii/hash-rstar:latest .
```
3. Run the container:
```bash
docker run -d -p 3000:3000 zhaoyii/hash-rstar:latest
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

