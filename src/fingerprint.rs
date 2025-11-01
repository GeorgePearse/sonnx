use blake3::Hasher;
use std::fmt::{self, Write};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Fingerprint([u8; 32]);

impl Fingerprint {
    pub fn from_hasher(hasher: Hasher) -> Self {
        Fingerprint(hasher.finalize().into())
    }

    pub fn derive(&self, tag: &[u8], index: u64) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(&self.0);
        hasher.update(tag);
        hasher.update(&index.to_le_bytes());
        Fingerprint::from_hasher(hasher)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(&self) -> String {
        let mut buf = String::with_capacity(64);
        for byte in self.0 {
            buf.write_fmt(format_args!("{:02x}", byte)).unwrap();
        }
        buf
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

pub struct FingerprintBuilder {
    hasher: Hasher,
}

impl FingerprintBuilder {
    pub fn new(tag: &[u8]) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(tag);
        FingerprintBuilder { hasher }
    }

    pub fn update_bytes(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
    }

    pub fn update_str(&mut self, value: &str) {
        self.update_bytes(value.as_bytes());
    }

    pub fn update_u64(&mut self, value: u64) {
        self.update_bytes(&value.to_le_bytes());
    }

    pub fn update_i64(&mut self, value: i64) {
        self.update_u64(value as u64);
    }

    pub fn update_f32(&mut self, value: f32) {
        self.update_bytes(&value.to_le_bytes());
    }

    pub fn finish(self) -> Fingerprint {
        Fingerprint::from_hasher(self.hasher)
    }
}
