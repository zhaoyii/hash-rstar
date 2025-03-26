use std::{fs::File, path::PathBuf, time::SystemTime};

use bincode::{Decode, Encode};
use hash_rstar::*;
use serde::Deserialize;

#[derive(Clone, PartialEq, Debug, Encode, Decode, Deserialize)]
#[serde(rename_all = "snake_case")]
struct Player {
    #[serde(rename = "uid_")]
    uid: String,
    name: String,
    #[serde(rename = "GCJ02_X")]
    x_coordinate: f64,
    #[serde(rename = "GCJ02_Y")]
    y_coordinate: f64,
}

impl Point for Player {
    fn point(&self) -> (f64, f64) {
        (self.x_coordinate, self.y_coordinate)
    }
}

impl Unique for Player {
    fn unique_id(&self) -> String {
        self.uid.clone()
    }
}

impl RstarPoint for Player {
    type Scalar = f64;
    const DIMENSIONS: usize = 2;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Player {
            uid: "".to_string(),
            name: "".to_string(),
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

fn main() {
    // let mut rdr = csv::Reader::from_reader(
    //     File::open("C:\\Users\\admin\\Desktop\\baidu_pois\\data3\\浙江+港澳.csv").unwrap(),
    // );
    let hrt: GeohashRTree<Player> = GeohashRTree::load(
        5,
        PathBuf::from("C:\\Users\\admin\\Desktop\\baidu_pois\\hash_rtree\\hash_rtree_db"),
    )
    .unwrap();

    let now = SystemTime::now();

    // let mut count = 0;
    // for result in rdr.deserialize() {
    //     let record: Player = result.unwrap();
    //     hrt.insert(record).unwrap();
    //     count += 1;
    //     if count % 10000 == 0 {
    //         println!(
    //             "load time: {:?}, len {} ",
    //             now.elapsed().unwrap(),
    //             hrt.len()
    //         );
    //         println!("load {} records", count);
    //     }
    // }

    println!(
        "load time: {:?}, len {} ",
        now.elapsed().unwrap(),
        hrt.len()
    );
}
