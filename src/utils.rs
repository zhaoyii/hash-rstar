use geo::{Coord, CoordsIter, Distance, Haversine, Line, Point};
use geohash::{decode_bbox, encode, neighbors};
use std::cmp::Ordering;

/// 计算点到直线的垂足
fn calculate_perpendicular_point(point: &Point<f64>, line: &Line<f64>) -> Point<f64> {
    let start = line.start;
    let end = line.end;

    // 如果起点和终点重合，返回起点
    if (start.x - end.x).abs() < f64::EPSILON && (start.y - end.y).abs() < f64::EPSILON {
        return Point::new(start.x, start.y);
    }

    // 计算线段向量
    let line_vector = Coord {
        x: end.x - start.x,
        y: end.y - start.y,
    };

    // 计算点到起点的向量
    let point_vector = Coord {
        x: point.x() - start.x,
        y: point.y() - start.y,
    };

    // 计算投影长度比例
    let t = (point_vector.x * line_vector.x + point_vector.y * line_vector.y)
        / (line_vector.x * line_vector.x + line_vector.y * line_vector.y);

    // 计算垂足坐标
    let foot_x = start.x + t * line_vector.x;
    let foot_y = start.y + t * line_vector.y;

    Point::new(foot_x, foot_y)
}

fn distance_to_line(point: &Point<f64>, line: &Line<f64>) -> f64 {
    let destination = calculate_perpendicular_point(point, line);
    Haversine::distance(point.clone(), destination)
}

fn find_nearest_point_distance(
    geohash_str: &str,
    point: &Point<f64>,
) -> Result<(Coord, f64), crate::GeohashRTreeError> {
    let rec = decode_bbox(geohash_str)?;
    let min = rec
        .coords_iter()
        .map(|c| {
            let distance = Haversine::distance(point.clone(), Point(c.clone()));
            (c, distance)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .unwrap();
    Ok(min)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Direction {
    /// Center
    C,
    /// North
    N,
    /// North-east
    NE,
    /// Eeast
    E,
    /// South-east
    SE,
    /// South
    S,
    /// South-west
    SW,
    /// West
    W,
    /// North-west
    NW,
}

/// Sorts geohash cells by minimum distance from a given point.
///
/// # Arguments
///
/// * `point` - The reference point to calculate distances from
/// * `geohash_precision` - The precision level of the geohash encoding
///
/// # Returns
///
/// Returns a `Result` containing a vector of tuples, each containing:
/// * A geohash string
/// * The minimum distance from the reference point
/// * The directional relationship (e.g., NE, SW, etc.)
///
/// The vector is sorted in ascending order by distance.
///
/// # Errors
///
/// Returns `GeohashRTreeError` if geohash operations fail.
pub fn sort_geohash_neighbors(
    point: Point<f64>,
    geohash_precision: usize,
) -> Result<Vec<(String, f64, Direction)>, crate::GeohashRTreeError> {
    let geohash_str = encode(Coord::from(point), geohash_precision)?;
    let nbs = neighbors(&geohash_str)?;

    let ne = find_nearest_point_distance(&nbs.ne, &point)?;
    let se = find_nearest_point_distance(&nbs.se, &point)?;
    let nw = find_nearest_point_distance(&nbs.nw, &point)?;
    let sw: (Coord, f64) = find_nearest_point_distance(&nbs.sw, &point)?;
    let e = distance_to_line(&point, &Line::new(ne.0.clone(), se.0.clone()));
    let s = distance_to_line(&point, &Line::new(se.0.clone(), sw.0.clone()));
    let w = distance_to_line(&point, &Line::new(nw.0.clone(), sw.0.clone()));
    let n = distance_to_line(&point, &Line::new(nw.0.clone(), ne.0.clone()));

    let mut sorted_neighbors = vec![
        (geohash_str, 0., Direction::C),
        (nbs.ne, ne.1, Direction::NE),
        (nbs.se, se.1, Direction::SE),
        (nbs.sw, sw.1, Direction::SW),
        (nbs.nw, nw.1, Direction::NW),
        (nbs.e, e, Direction::E),
        (nbs.s, s, Direction::S),
        (nbs.w, w, Direction::W),
        (nbs.n, n, Direction::N),
    ];

    sorted_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    Ok(sorted_neighbors)
}

#[cfg(test)]
mod tests {

    use super::*;
    use geo::{Line, Point};

    #[test]
    fn test_perpendicular_point() {
        // 测试水平线
        let point = Point::new(1.0, 1.0);
        let line = Line::new(Point::new(0.0, 0.0), Point::new(2.0, 0.0));
        let foot = calculate_perpendicular_point(&point, &line);
        assert!((foot.x() - 1.0).abs() < f64::EPSILON);
        assert!(foot.y().abs() < f64::EPSILON);

        // 测试垂直线
        let point2 = Point::new(1.0, 1.0);
        let line2 = Line::new(Point::new(0.0, 0.0), Point::new(0.0, 2.0));
        let foot2 = calculate_perpendicular_point(&point2, &line2);
        assert!(foot2.x().abs() < f64::EPSILON);
        assert!((foot2.y() - 1.0).abs() < f64::EPSILON);

        // 测试45度斜线
        let point3 = Point::new(0.0, 1.0);
        let line3 = Line::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        let foot3 = calculate_perpendicular_point(&point3, &line3);
        assert!((foot3.x() - 0.5).abs() < f64::EPSILON);
        assert!((foot3.y() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sort_geohash_neighbors() {
        let point = Point::new(114.432292, 30.545003);
        let geohash_precision = 5usize;
        let neighbors = sort_geohash_neighbors(point, geohash_precision).unwrap();
        assert_eq!(neighbors[0], ("wt3mg".into(), 0.0, Direction::C));
        assert_eq!(neighbors.last().unwrap().0, "wt3q4");
        assert_eq!(neighbors.last().unwrap().2, Direction::NW);
    }
}
