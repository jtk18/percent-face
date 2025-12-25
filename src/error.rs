use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model deserialization error: {0}")]
    Deserialization(#[from] bincode::Error),

    #[error("Invalid model: {0}")]
    InvalidModel(String),

    #[error("Image access error: pixel ({x}, {y}) out of bounds for {width}x{height} image")]
    PixelOutOfBounds {
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    },
}

pub type Result<T> = std::result::Result<T, Error>;
