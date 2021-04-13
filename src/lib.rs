//! The `redisai-rs` provide a rust client to interact with [RedisAI](https://oss.redislabs.com/redisai/).
//!
//! Checkout the documentation for API details and examples of the Client.

/// To use the RedisAIClient you need to create a redis-rs client.
/// [redis-rs](https://docs.rs/redis/0.20.0/redis/index.html)
/// ```
/// use redisai::{RedisAIClient, AIDataType};
/// use redis::Client;
/// let aiclient: RedisAIClient = RedisAIClient { debug: false };
/// let client = Client::open("redis://127.0.0.1/").unwrap();
/// let mut con = client.get_connection().unwrap();
///
/// let tensor: Vec<f64> = vec![1., 2., 3., 4.];
/// let shape: Vec<usize> = vec![4];
///     aiclient.ai_tensorset(
///         &mut con,
///         "one_dim_double_tensor".to_string(),
///         AIDataType::DOUBLE,
///          shape,
///          tensor
///     );
/// ```
#[derive(Debug, Clone)]
pub struct RedisAIClient {
    pub debug: bool,
}

///Available datatype in this crate
#[derive(Debug, PartialEq, Clone, strum_macros::EnumString, strum_macros::ToString)]
pub enum AIDataType {
    FLOAT,
    DOUBLE,
    INT8,
    INT16,
    INT32,
    INT64,
    UNIT8,
    UNIT16,
}

/// Documentation for the config interface
pub mod config;
/// Documentation for the info interface
pub mod info;
/// Documentation for the tensor interface
pub mod tensor;
