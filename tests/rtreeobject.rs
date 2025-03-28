use rstar::{AABB, RTree, PointDistance, RTreeObject};
use geo::{Distance, Haversine};

#[derive(Debug)]
struct Player {
    name: String,
    x_coordinate: f64,
    y_coordinate: f64,
}

impl RTreeObject for Player {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.x_coordinate, self.y_coordinate])
    }
}

// Implement PointDistance for Player
impl PointDistance for Player {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let self_geo_point = geo::point!(x: self.x_coordinate, y: self.y_coordinate);
        let target_geo_point = geo::point!(x: point[0], y: point[1]);
        Haversine::distance(self_geo_point, target_geo_point)
    }
}

#[test]
fn test_rtree_object() {
    let mut tree = RTree::new();
    // Insert a few players...
    tree.insert(Player {
        name: "Forlorn Freeman".into(),
        x_coordinate: 1.,
        y_coordinate: 0.,
    });
    tree.insert(Player {
        name: "Sarah Croft".into(),
        x_coordinate: 0.5,
        y_coordinate: 0.5,
    });
    tree.insert(Player {
        name: "Geralt of Trivia".into(),
        x_coordinate: 0.,
        y_coordinate: 2.,
    });

    // Now we are ready to ask some questions!
    let envelope = AABB::from_point([0.5, 0.5]);
    let likely_sarah_croft = tree.locate_in_envelope(&envelope).next();
    println!(
        "Found {:?} lurking around at (0.5, 0.5)!",
        likely_sarah_croft.unwrap().name
    );


    let unit_square = AABB::from_corners([-1.0, -1.0], [1., 1.]);
    for player in tree.locate_in_envelope(&unit_square) {
        println!(
            "And here is {:?} spelunking in the unit square.",
            player.name
        );
    }

    println!("{:?}", tree.nearest_neighbor(&[0.5, 0.5]).unwrap());
}
