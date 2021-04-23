#![crate_name = "redisai"]
//! The `redisai-rs` provide a rust client to interact with [RedisAI](https://oss.redislabs.com/redisai/).
//!
//! Checkout the documentation for API details and examples of the Client.
/// To use the RedisAIClient you need to create a redis-rs client.
/// [redis-rs](https://docs.rs/redis/0.20.0/redis/index.html)
/// ```
/// // Using the redis crate to create a client and connection
/// use redis::Client;
///
/// // Importing our custom types
/// use redisai::{RedisAIClient, AIDataType};
/// use redisai::tensor::AITensor;
///
/// let client = Client::open("redis://127.0.0.1/").unwrap();
/// let mut con = client.get_connection().unwrap();
///
/// let aiclient: RedisAIClient = RedisAIClient { debug: true };
///
/// let tensor_data: Vec<u8> = vec![1, 2, 3, 4];
/// let ai_tensor: AITensor<u8, 1> = AITensor::new([4], tensor_data);
///     aiclient.ai_tensorset(
///         &mut con,
///         "one_dim_vec_tensor".to_string(),
///          ai_tensor
///     );
/// ```
use serde::{Deserialize, Serialize};
use std::convert::From;
/// Main struct of the crate. I will be the support for the implementation of the commands.
#[derive(Debug, Clone)]
pub struct RedisAIClient {
    /// Turn this on to echo the command to stdout
    pub debug: bool,
}

/// Available datatype in this crate
#[derive(
    Debug,
    Deserialize,
    Serialize,
    PartialEq,
    Clone,
    strum_macros::EnumString,
    strum_macros::ToString,
)]
pub enum AIDataType {
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    FLOAT,
    DOUBLE,
}
/// Trait for converting rust primitive type to AIDataType.
pub trait ToAIDataType {
    fn to_aidtype() -> AIDataType;
}
macro_rules! impl_from_AIDataType {
    ($inner_type:ty, $dtype:expr) => {
        impl ToAIDataType for $inner_type {
            fn to_aidtype() -> AIDataType {
                $dtype
            }
        }
    };
}

impl_from_AIDataType! {i8, AIDataType::INT8}
impl_from_AIDataType! {i16, AIDataType::INT16}
impl_from_AIDataType! {i32, AIDataType::INT32}
impl_from_AIDataType! {i64, AIDataType::INT64}
impl_from_AIDataType! {u8, AIDataType::UINT8}
impl_from_AIDataType! {u16, AIDataType::UINT16}
impl_from_AIDataType! {f32, AIDataType::FLOAT}
impl_from_AIDataType! {f64, AIDataType::DOUBLE}

/// Available backend for this crate
#[derive(
    Debug,
    Deserialize,
    Serialize,
    PartialEq,
    Clone,
    strum_macros::EnumString,
    strum_macros::ToString,
)]
pub enum Backend {
    TF,
    TFLITE,
    TORCH,
    ONNX,
}
/// Available device for this crate
#[derive(
    Debug,
    Deserialize,
    Serialize,
    PartialEq,
    Clone,
    strum_macros::EnumString,
    strum_macros::ToString,
)]
pub enum Device {
    // Default device if not specify
    CPU,
    // Default to GPU:0 if used but can be customize to any available gpu
    GPU(usize),
}

/// Configuration directives at run-time for RedisAI module
pub mod config;
/// General module information or information about the execution a model
pub mod info;

/// Upload and launch model execution
pub mod model;
/// Upload and retrieve tensors
pub mod tensor;
