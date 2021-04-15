# redisai-rs

A rust client for [RedisAI](https://oss.redislabs.com/redisai)

Please read the [API documentation on doc.rs](https://docs.rs/redisai)

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/) ![Crates.io](https://img.shields.io/crates/v/redisai?label=crates.io) ![docs.rs](https://img.shields.io/docsrs/redisai) ![GitHub Workflow Status](https://img.shields.io/github/workflow/status/thomaub/redisai-rs/General%20Rust%20CI)

## Installation

The RedisAI module needs to be installed first.
A docker image is provided for convinience from redislab:

```bash
# cpu version
docker run -p 6379:6379 -it --rm redislabs/redisai:edge-cpu
# gpu version
docker run -p 6379:6379 -it --rm redislabs/redisai:edge-gpu
```

The gpu version use nvidia runtime so nvidia gpu needto be available.  More information can be found in [RedisAI github](https://github.com/RedisAI/RedisAI)

Then simply add it to your `Cargo.toml`

```toml
[dependencies]
redis = "0.20.0"
redisai = "0.1.1"
```

## Usage

```rust
use redis;

use redisai::RedisAIClient;
use redisai::tensor::{ToFromBlob,AITensor};

let aiclient: RedisAIClient = RedisAIClient { debug: true };
let client = redis::Client::open("redis://127.0.0.1/").unwrap();
let mut con = client.get_connection().unwrap();

let tensor_data: Vec<i8> = vec![1, 2, 3, 127];
let shape: [usize; 1] = [4];
let ai_tensor: AITensor<i8, 1> = AITensor::new(shape, tensor_data.to_blob());

aiclient.ai_tensorset(&mut con, "example_one_dim_i8_tensor".to_string(), ai_tensor);

let ai_tensor: AITensor<i8, 1> = aiclient.ai_tensorget(&mut con, "example_one_dim_i8_tensor".to_string(), false).unwrap();
```
