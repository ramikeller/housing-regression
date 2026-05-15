mod dataset;

fn main() {
    let (train, test, _normalizer) = dataset::load("data/housing.csv");
    println!("Train rows: {}", train.len());
    println!("Test rows:  {}", test.len());
    println!("First train feature vector: {:?}", train[0].features);
    println!("First train target:         {:.4}", train[0].target);
}
