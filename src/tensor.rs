use crate::AIDataType;
/// Documentation for the tensor interface
use crate::RedisAIClient;
use redis::FromRedisValue;
use redis::{RedisResult, Value};
use std::fmt;
use std::str::FromStr;
use std::string::ToString;

pub trait ToFromBlob {
    fn to_blob(&self) -> Vec<u8>;
    fn from_blob(blob: &[u8]) -> Self;
}

macro_rules! impl_tofrom_blob {
    ($inner_type:ty, $size:expr) => {
        impl ToFromBlob for Vec<$inner_type> {
            fn to_blob(&self) -> Vec<u8> {
                self.iter()
                    .flat_map(|elem| { elem.to_be_bytes().to_vec().into_iter() }.into_iter())
                    .collect::<Vec<u8>>()
            }
            fn from_blob(blob: &[u8]) -> Self {
                blob.chunks_exact($size)
                    .map(|bytes| {
                        let mut arr = [0u8; $size];
                        arr.copy_from_slice(&bytes[0..$size]);
                        <$inner_type>::from_be_bytes(arr)
                    })
                    .collect()
            }
        }
    };
}
impl_tofrom_blob! {i8, 1}
impl_tofrom_blob! {i16, 2}
impl_tofrom_blob! {i32, 4}
impl_tofrom_blob! {i64, 8}
impl_tofrom_blob! {u8, 1}
impl_tofrom_blob! {u16, 2}
impl_tofrom_blob! {f32, 4}
impl_tofrom_blob! {f64, 8}

#[derive(Debug, PartialEq)]
pub struct AITensorMeta {
    dtype: AIDataType,
    shape: Vec<usize>,
}
#[derive(Debug, PartialEq)]
pub struct AITensor<T> {
    meta: AITensorMeta,
    blob: T,
}

// TODO: remove unwrap() ?
impl<T> FromRedisValue for AITensor<T>
where
    T: fmt::Debug + FromRedisValue + ToFromBlob,
{
    fn from_redis_value(v: &Value) -> RedisResult<Self> {
        let mut it = v.as_sequence().unwrap().iter();
        let meta: AITensorMeta = match (it.next(), it.next(), it.next(), it.next()) {
            (
                _, //Some(dtype_field),
                Some(dtype),
                _, //Some(shape_field),
                Some(shape),
            ) => AITensorMeta {
                dtype: AIDataType::from_str(&String::from_redis_value(dtype)?).unwrap(),
                shape: Vec::<usize>::from_redis_value(shape).unwrap(),
            },

            _ => {
                return Err(redis::RedisError::from((
                    redis::ErrorKind::TypeError,
                    "Cannot convert AITensorMeta field to some string",
                )))
            }
        };

        let blob = match (it.next(), it.next()) {
            (
                _, //Some(blob_field),
                Some(blob),
            ) => {
                let blob_bytes = Vec::<u8>::from_redis_value(blob).unwrap();
                <T>::from_blob(blob_bytes.as_slice())
                // let blob_correct_type = match meta.dtype {
                //     AIDataType::FLOAT => {}
                //     AIDataType::DOUBLE => {}
                //     AIDataType::INT8 => {}
                //     AIDataType::INT16 => {}
                //     AIDataType::INT32 => {}
                //     AIDataType::INT64 => {}
                //     AIDataType::UNIT8 => {}
                //     AIDataType::UNIT16 => {}
                // }
            }
            _ => {
                return Err(redis::RedisError::from((
                    redis::ErrorKind::TypeError,
                    "Cannot convert AITensor blob field to some string",
                )))
            }
        };

        Ok(Self { meta, blob })
    }
}

impl RedisAIClient {
    pub fn ai_tensorset<T>(
        &self,
        con: &mut redis::Connection,
        key: String,
        dtype: AIDataType,
        shape: Vec<usize>,
        tensor: Vec<T>,
    ) -> RedisResult<()>
    where
        Vec<T>: ToFromBlob,
        // Maybe possible with just `to_be_bytes` and no custom Trait but don't know how.
        // TODO: Follow-up on std::num::Trait https://github.com/rust-num/num-traits/pull/103/
    {
        let dtype_str = dtype.to_string();
        let shape_str = shape.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        let blob = tensor.to_blob();
        if self.debug {
            println!(
                "AI.TENSORSET {} {} {:?} BLOB {:#04X?}",
                &key, &dtype_str, &shape_str, &blob
            );
        }
        redis::cmd("AI.TENSORSET")
            .arg(key)
            .arg(dtype_str)
            .arg(shape_str)
            .arg("BLOB")
            .arg(blob)
            .query(con)?;
        Ok(())
    }
    pub fn ai_tensorget<T>(
        &self,
        con: &mut redis::Connection,
        key: String,
    ) -> RedisResult<AITensor<T>>
    where
        T: fmt::Debug + FromRedisValue + ToFromBlob,
    {
        if self.debug {
            println!("AI.TENSORGET {:?} META BLOB", &key);
        }

        let tensor: AITensor<T> = redis::cmd("AI.TENSORGET")
            .arg(key)
            .arg("META")
            .arg("BLOB")
            .query(con)?;
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn ai_tensorset_one_dim_int8() {
        let aiclient: RedisAIClient = RedisAIClient { debug: true };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();
        let tensor: Vec<i8> = vec![1, 2, 3, 4];
        let shape: Vec<usize> = vec![4];
        assert_eq!(
            Ok(()),
            aiclient.ai_tensorset(
                &mut con,
                "one_dim_i8_tensor".to_string(),
                AIDataType::INT8,
                shape,
                tensor
            )
        );
    }
    #[test]
    fn ai_tensorget_one_dim_int8() {
        let aiclient: RedisAIClient = RedisAIClient { debug: false };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();
        let tensor_data: Vec<i8> = vec![0x01, 0x02, 0x03, 0x04];
        let shape: Vec<usize> = vec![4];
        assert_eq!(
            Ok(AITensor {
                meta: AITensorMeta {
                    dtype: AIDataType::INT8,
                    shape: shape
                },
                blob: tensor_data
            }),
            aiclient.ai_tensorget(&mut con, "one_dim_i8_tensor".to_string())
        );
    }
    #[test]
    fn ai_tensorset_three_dim_int32() {
        let aiclient: RedisAIClient = RedisAIClient { debug: false };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();
        let tensor: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let shape: Vec<usize> = vec![2, 2, 3];
        assert_eq!(
            Ok(()),
            aiclient.ai_tensorset(
                &mut con,
                "three_dim_i32_tensor".to_string(),
                AIDataType::INT32,
                shape,
                tensor
            )
        );
    }
    #[test]
    fn ai_tensorget_three_dim_int32() {
        let aiclient: RedisAIClient = RedisAIClient { debug: false };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();
        let tensor_data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let shape: Vec<usize> = vec![2, 2, 3];
        assert_eq!(
            Ok(AITensor {
                meta: AITensorMeta {
                    dtype: AIDataType::INT32,
                    shape: shape
                },
                blob: tensor_data
            }),
            aiclient.ai_tensorget(&mut con, "three_dim_i32_tensor".to_string())
        );
    }
    #[test]
    fn ai_tensorset_one_dim_float32() {
        let aiclient: RedisAIClient = RedisAIClient { debug: false };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();
        let tensor: Vec<f32> = vec![1., 2., 3., 4.];
        let shape: Vec<usize> = vec![4];
        assert_eq!(
            Ok(()),
            aiclient.ai_tensorset(
                &mut con,
                "one_dim_f32_tensor".to_string(),
                AIDataType::FLOAT,
                shape,
                tensor
            )
        );
    }
    #[test]
    fn ai_tensorget_one_dim_float32() {
        let aiclient: RedisAIClient = RedisAIClient { debug: false };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();
        let tensor_data: Vec<f32> = vec![1., 2., 3., 4.];
        let shape: Vec<usize> = vec![4];
        assert_eq!(
            Ok(AITensor {
                meta: AITensorMeta {
                    dtype: AIDataType::FLOAT,
                    shape: shape
                },
                blob: tensor_data
            }),
            aiclient.ai_tensorget(&mut con, "one_dim_f32_tensor".to_string())
        );
    }
    #[test]
    fn ai_tensorset_three_dim_float32() {
        let aiclient: RedisAIClient = RedisAIClient { debug: false };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();
        let tensor: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let shape: Vec<usize> = vec![2, 2, 3];
        assert_eq!(
            Ok(()),
            aiclient.ai_tensorset(
                &mut con,
                "three_dim_f32_tensor".to_string(),
                AIDataType::FLOAT,
                shape,
                tensor
            )
        );
    }
    #[test]
    fn ai_tensorget_three_dim_float32() {
        let aiclient: RedisAIClient = RedisAIClient { debug: false };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();
        let tensor_data: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let shape: Vec<usize> = vec![2, 2, 3];
        assert_eq!(
            Ok(AITensor {
                meta: AITensorMeta {
                    dtype: AIDataType::FLOAT,
                    shape: shape
                },
                blob: tensor_data
            }),
            aiclient.ai_tensorget(&mut con, "three_dim_f32_tensor".to_string())
        );
    }
    #[test]
    fn ai_tensorset_one_dim_float64() {
        let aiclient: RedisAIClient = RedisAIClient { debug: false };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();
        let tensor: Vec<f64> = vec![1., 2., 3., 4.];
        let shape: Vec<usize> = vec![4];
        assert_eq!(
            Ok(()),
            aiclient.ai_tensorset(
                &mut con,
                "one_dim_double_tensor".to_string(),
                AIDataType::DOUBLE,
                shape,
                tensor
            )
        );
    }
    #[test]
    fn ai_tensorget_one_dim_float64() {
        let aiclient: RedisAIClient = RedisAIClient { debug: false };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();
        let tensor_data: Vec<f64> = vec![1., 2., 3., 4.];
        let shape: Vec<usize> = vec![4];
        assert_eq!(
            Ok(AITensor {
                meta: AITensorMeta {
                    dtype: AIDataType::DOUBLE,
                    shape: shape
                },
                blob: tensor_data
            }),
            aiclient.ai_tensorget(&mut con, "one_dim_double_tensor".to_string())
        );
    }
    #[test]
    fn ai_tensorset_from_2d_ndarray() {
        let aiclient: RedisAIClient = RedisAIClient { debug: false };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();
        let tensor: ndarray::Array2<f32> = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        let shape: Vec<usize> = vec![2, 3];
        assert_eq!(
            Ok(()),
            aiclient.ai_tensorset(
                &mut con,
                "two_dim_double_ndarray_tensor".to_string(),
                AIDataType::FLOAT,
                shape,
                tensor.iter().map(|elem| elem.clone()).collect::<Vec<f32>>()
            )
        );
    }
}
