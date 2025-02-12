use rand::rngs::StdRng;
use rand::RngCore;
use rand::SeedableRng;

pub fn get_rng(seed: Option<u64>) -> impl RngCore {
    match seed {
        Some(state) => StdRng::seed_from_u64(state),
        None => StdRng::from_entropy(),
    }
}
