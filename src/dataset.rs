use serde::Deserialize;

// One struct field per CSV column. Option<f64> on total_bedrooms because
// some rows have a blank there — serde will parse those as None.
#[derive(Debug, Deserialize)]
pub struct HousingRecord {
    pub longitude: f64,
    pub latitude: f64,
    pub housing_median_age: f64,
    pub total_rooms: f64,
    pub total_bedrooms: Option<f64>,
    pub population: f64,
    pub households: f64,
    pub median_income: f64,
    pub median_house_value: f64,
    pub ocean_proximity: String, // text column — we'll ignore it for now
}

// A cleaned row with all 8 numeric features and the target value.
#[derive(Debug, Clone)]
pub struct HousingRow {
    pub features: [f64; 8],   // inputs to the model
    pub target: f64,           // what we want the model to predict
}

// Stores the per-feature min and max so we can normalize consistently.
pub struct Normalizer {
    pub mins: [f64; 9],  // 8 features + 1 target
    pub maxs: [f64; 9],
}

impl Normalizer {
    // Compute min/max from the training rows (before normalization).
    pub fn from_rows(rows: &[HousingRow]) -> Self {
        let mut mins = [f64::MAX; 9];
        let mut maxs = [f64::MIN; 9];

        for row in rows {
            for i in 0..8 {
                mins[i] = mins[i].min(row.features[i]);
                maxs[i] = maxs[i].max(row.features[i]);
            }
            mins[8] = mins[8].min(row.target);
            maxs[8] = maxs[8].max(row.target);
        }

        Self { mins, maxs }
    }

    // Scale a single value from column `i` into [0, 1].
    pub fn scale(&self, value: f64, i: usize) -> f64 {
        let range = self.maxs[i] - self.mins[i];
        if range == 0.0 { 0.0 } else { (value - self.mins[i]) / range }
    }

    // Apply normalization to every feature and the target in a row.
    pub fn normalize(&self, row: &HousingRow) -> HousingRow {
        let mut features = [0.0f64; 8];
        for i in 0..8 {
            features[i] = self.scale(row.features[i], i);
        }
        HousingRow {
            features,
            target: self.scale(row.target, 8),
        }
    }
}

// Reads the CSV, drops incomplete rows, and returns (train_rows, test_rows, normalizer).
// The split is 80% train / 20% test.
pub fn load(path: &str) -> (Vec<HousingRow>, Vec<HousingRow>, Normalizer) {
    let mut reader = csv::Reader::from_path(path).unwrap_or_else(|_| panic!("could not open {path}"));

    let mut rows: Vec<HousingRow> = Vec::new();
    for result in reader.deserialize::<HousingRecord>() {
        let record = result.expect("CSV parse error");

        // Skip rows with missing total_bedrooms.
        let total_bedrooms = match record.total_bedrooms {
            Some(v) => v,
            None => continue,
        };

        rows.push(HousingRow {
            features: [
                record.longitude,
                record.latitude,
                record.housing_median_age,
                record.total_rooms,
                total_bedrooms,
                record.population,
                record.households,
                record.median_income,
            ],
            target: record.median_house_value,
        });
    }

    // Split before normalizing — normalizer must only see training data.
    let split = (rows.len() as f64 * 0.8) as usize;
    let test_raw = rows.split_off(split);
    let train_raw = rows;

    let normalizer = Normalizer::from_rows(&train_raw);

    let train: Vec<HousingRow> = train_raw.iter().map(|r| normalizer.normalize(r)).collect();
    let test: Vec<HousingRow> = test_raw.iter().map(|r| normalizer.normalize(r)).collect();

    (train, test, normalizer)
}
