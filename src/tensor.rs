/// Documentation for the tensor module
use crate::{AIDataType, RedisAIClient, ToAIDataType};

use std::convert::From;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use std::string::ToString;

use redis::FromRedisValue;
use redis::{RedisResult, Value};

#[cfg(feature = "ndar")]
use ndarray::{Array, Dimension};
/// Trait to tansform a generic container type like a Vec<f32> or ndarray to a Vec<u8> used for BLOB
pub trait ToFromBlob {
    fn to_blob(self) -> Vec<u8>;
    fn from_blob(blob: &[u8], shape: &[usize]) -> Self;
}

macro_rules! impl_tofrom_blob_vec {
    ($inner_type:ty) => {
        impl ToFromBlob for Vec<$inner_type> {
            fn to_blob(self) -> Vec<u8> {
                self.iter()
                    .flat_map(|elem| { elem.to_be_bytes().to_vec().into_iter() }.into_iter())
                    .collect::<Vec<u8>>()
            }
            fn from_blob(blob: &[u8], _shape: &[usize]) -> Self {
                const SIZE: usize = std::mem::size_of::<$inner_type>();
                blob.chunks_exact(SIZE)
                    .map(|bytes| {
                        let mut arr = [0u8; SIZE];
                        arr.copy_from_slice(&bytes[0..SIZE]);
                        <$inner_type>::from_be_bytes(arr)
                    })
                    .collect()
            }
        }
    };
}
impl_tofrom_blob_vec! {i8}
impl_tofrom_blob_vec! {i16}
impl_tofrom_blob_vec! {i32}
impl_tofrom_blob_vec! {i64}
impl_tofrom_blob_vec! {u8}
impl_tofrom_blob_vec! {u16}
impl_tofrom_blob_vec! {f32}
impl_tofrom_blob_vec! {f64}

#[cfg(feature = "ndar")]
impl<S, D, const N: usize> From<Array<S, D>> for AITensor<S, N>
where
    S: Debug + ToAIDataType,
    Vec<S>: ToFromBlob,
    D: Dimension,
{
    fn from(ndarray: Array<S, D>) -> Self {
        let shape_slice = ndarray.shape();
        let mut shape: [usize; N] = [0; N];
        let size = shape_slice.len();
        if shape_slice.len() != N {
            panic!(
                "impossible to convert a {} ndarray into an {} AITensor",
                size, N
            )
        }
        for i in 0..N {
            shape[i] = shape_slice[i]
        }
        AITensor {
            meta: AITensorMeta {
                dtype: S::to_aidtype(),
                shape,
                phantom: PhantomData,
            },
            blob: ndarray.into_raw_vec().to_blob(),
        }
    }
}

/// The representation of the metadata of our tensor.
#[derive(Debug, PartialEq, Clone)]
pub struct AITensorMeta<T, const S: usize> {
    /// The tensor's data type
    pub dtype: AIDataType,
    /// The tensor's shape and S represent the dimension
    pub shape: [usize; S],
    phantom: PhantomData<T>,
}

impl<T, const S: usize> AITensorMeta<T, S>
where
    T: Debug + ToAIDataType,
{
    pub fn new(shape: [usize; S]) -> Self {
        Self {
            dtype: T::to_aidtype(),
            shape: shape,
            phantom: PhantomData,
        }
    }
}
/// The tensor that represents an n-dimensional array of values
#[derive(Debug, PartialEq, Clone)]
pub struct AITensor<T, const S: usize> {
    /// The metadata of the tensor
    pub meta: AITensorMeta<T, S>,
    /// The blob representation of the tensor values
    pub blob: Vec<u8>,
}

impl<T, const S: usize> AITensor<T, S>
where
    T: Debug + ToAIDataType,
{
    pub fn new(shape: [usize; S], blob: Vec<u8>) -> Self {
        let meta: AITensorMeta<T, S> = AITensorMeta {
            dtype: T::to_aidtype(),
            shape: shape,
            phantom: PhantomData,
        };
        Self { meta, blob }
    }
}

// TODO: remove unwrap() ?
// TODO: remove duplicated code for Meta and Tensor
impl<T, const S: usize> FromRedisValue for AITensorMeta<T, S>
where
    T: Debug,
    Vec<T>: ToFromBlob,
{
    fn from_redis_value(v: &Value) -> RedisResult<Self> {
        let mut it = v.as_sequence().unwrap().iter();
        let meta: AITensorMeta<T, S> = match (it.next(), it.next(), it.next(), it.next()) {
            (
                _, //Some(dtype_field),
                Some(dtype),
                _, //Some(shape_field),
                Some(shape),
            ) => {
                let shape_vec = Vec::<usize>::from_redis_value(shape).unwrap();
                if shape_vec.len() != S {
                    return Err(redis::RedisError::from((
                        redis::ErrorKind::TypeError,
                        "Size of the retrieve data shape doesn't match the expected const shape",
                    )));
                }
                let mut shape_arr: [usize; S] = [0; S];
                for i in 0..S {
                    shape_arr[i] = shape_vec[i];
                }
                AITensorMeta {
                    dtype: AIDataType::from_str(&String::from_redis_value(dtype)?).unwrap(),
                    shape: shape_arr,
                    phantom: PhantomData,
                }
            }
            _ => {
                return Err(redis::RedisError::from((
                    redis::ErrorKind::TypeError,
                    "Cannot convert AITensorMeta field to some string",
                )))
            }
        };
        Ok(meta)
    }
}
impl<T, const S: usize> FromRedisValue for AITensor<T, S>
where
    T: Debug,
    Vec<T>: ToFromBlob,
{
    fn from_redis_value(v: &Value) -> RedisResult<Self> {
        let it = v.as_sequence().unwrap().iter();
        let meta_part_iter = it.clone().take(4);
        let meta_part = Value::Bulk(
            meta_part_iter
                .map(|elem| elem.clone())
                .collect::<Vec<Value>>(),
        );
        let meta: AITensorMeta<T, S> = AITensorMeta::<T, S>::from_redis_value(&meta_part).unwrap();

        let mut blob_part_iter = it.skip(4);
        let blob = match (blob_part_iter.next(), blob_part_iter.next()) {
            (Some(blob_field), Some(blob)) => {
                dbg!(&blob_field);
                dbg!(&blob);
                let blob_bytes = Vec::<u8>::from_redis_value(blob).unwrap();
                blob_bytes
            }
            _ => {
                return Err(redis::RedisError::from((
                    redis::ErrorKind::TypeError,
                    "Cannot convert AITensor blob field to some string",
                )))
            }
        };
        let data_size: usize =
            meta.shape.iter().fold(1, |acc, elem| acc * elem) * std::mem::size_of::<T>();
        dbg!(&data_size);
        // Sanity check for the data
        if blob.len() != data_size {
            return Err(redis::RedisError::from((
                redis::ErrorKind::TypeError,
                "Size of the retrieve data doesn't match the expected shape",
            )));
        }

        Ok(Self { meta, blob })
    }
}

fn modelset_cmd_build<T, const S: usize>(key: String, tensor: &AITensor<T, S>) -> Vec<String> {
    let mut args_command: Vec<String> = vec![key];
    args_command.push(tensor.meta.dtype.to_string());
    args_command.append(
        &mut tensor
            .meta
            .shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>(),
    );
    args_command
}
fn modelget_cmd_build(key: String, meta_only: bool) -> Vec<String> {
    let mut args_command: Vec<String> = vec![key, "META".to_string()];
    if meta_only {
    } else {
        args_command.push("BLOB".to_string());
    };
    args_command
}

impl RedisAIClient {
    /// The [AI.TENSORSET](https://oss.redislabs.com/redisai/commands/#aitensorset) command
    /// It stores a tensor as the value of a key.
    ///
    /// **Note**: The implementation differ form the raw AI.TENSORSET command.
    /// It's currently not possible to create an uninitialized tensor.
    ///
    /// Only tensors with actual data can be sent to redis so BLOB sub command is always called.
    ///
    /// Also the VALUES sub command is not used.
    ///
    /// From RedisAI docs:
    /// > While it is possible to set the tensor using binary data or numerical values,
    /// > it is recommended that you use the BLOB option.
    /// > It requires fewer resources and performs better compared to specifying the values discretely.
    /// ```
    /// use redis;
    /// use redisai::RedisAIClient;
    /// use redisai::tensor::{ToFromBlob,AITensor};
    ///
    /// let aiclient: RedisAIClient = RedisAIClient { debug: true };
    /// let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    /// let mut con = client.get_connection().unwrap();
    ///
    /// let tensor_data: Vec<i8> = vec![1, 2, 3, 127];
    /// let shape: [usize; 1] = [4];
    /// let ai_tensor: AITensor<i8, 1> = AITensor::new(shape, tensor_data.to_blob());
    /// aiclient.ai_tensorset(&mut con, "example_one_dim_i8_tensor".to_string(), ai_tensor);
    /// ```
    pub fn ai_tensorset<T, const S: usize>(
        &self,
        con: &mut redis::Connection,
        key: String,
        tensor: AITensor<T, S>,
    ) -> RedisResult<()>
    where
        T: Debug,
        // Maybe possible with just `to_be_bytes` and no custom Trait but don't know how.
        // TODO: Follow-up on std::num::Trait https://github.com/rust-num/num-traits/pull/103/
    {
        let args = modelset_cmd_build(key, &tensor);
        if self.debug {
            println!("AI.TENSORSET {:?} BLOB {:#04X?}", args, &tensor.blob);
        }
        redis::cmd("AI.TENSORSET")
            .arg(args)
            .arg("BLOB")
            .arg(tensor.blob)
            .query(con)?;
        Ok(())
    }
    /// The [AI.TENSORGET](https://oss.redislabs.com/redisai/commands/#aitensorget) command
    /// It returns a tensor stored as key's value.
    ///
    /// meta_only is true -> only the AITensorMeta of the AITensor is correct. The blob field is an empty Vec
    ///
    /// meta_only is false -> the complete AITensor is returned and in the BLOB format.
    /// VALUES format not supported.
    /// _It's not possible to only retrieve the BLOB because we want the META to enforce the return type_
    ///
    /// ```
    /// use redis;
    /// use redisai::RedisAIClient;
    /// use redisai::tensor::{ToFromBlob,AITensor};
    ///
    /// let aiclient: RedisAIClient = RedisAIClient { debug: true };
    /// let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    /// let mut con = client.get_connection().unwrap();
    ///
    /// let tensor_data: Vec<i8> = vec![1, 2, 3, 127];
    /// let shape: [usize; 1] = [4];
    /// let ai_tensor: AITensor<i8, 1> = AITensor::new(shape, tensor_data.to_blob());
    /// aiclient.ai_tensorset(&mut con, "example_one_dim_i8_tensor".to_string(), ai_tensor);
    ///
    /// let ai_tensor: AITensor<i8, 1> = aiclient.ai_tensorget(&mut con, "example_one_dim_i8_tensor".to_string(), false).unwrap();
    /// println!("{:?}", ai_tensor);
    /// ```
    pub fn ai_tensorget<T, const S: usize>(
        &self,
        con: &mut redis::Connection,
        key: String,
        meta_only: bool,
    ) -> RedisResult<AITensor<T, S>>
    where
        T: Debug,
        Vec<T>: ToFromBlob,
    {
        let args = modelget_cmd_build(key, meta_only);
        let tensor = if meta_only {
            if self.debug {
                println!("AI.TENSORGET {:?}", args)
            }
            let meta: AITensorMeta<T, S> = redis::cmd("AI.TENSORGET").arg(args).query(con)?;
            AITensor { meta, blob: vec![] }
        } else {
            let tensor: AITensor<T, S> = redis::cmd("AI.TENSORGET").arg(args).query(con)?;
            tensor
        };
        Ok(tensor)
    }
}

#[cfg(feature = "aio")]
impl RedisAIClient {
    pub async fn ai_tensorset_async<T, const S: usize>(
        &self,
        con: &mut redis::aio::Connection,
        key: String,
        tensor: AITensor<T, S>,
    ) -> RedisResult<()>
    where
        T: Debug,
    {
        let args = modelset_cmd_build(key, &tensor);
        if self.debug {
            println!("AI.TENSORSET {:?} BLOB {:#04X?}", args, &tensor.blob);
        }
        redis::cmd("AI.TENSORSET")
            .arg(args)
            .arg("BLOB")
            .arg(tensor.blob)
            .query_async(con)
            .await?;
        Ok(())
    }
    pub async fn ai_tensorget_async<T, const S: usize>(
        &self,
        con: &mut redis::aio::Connection,
        key: String,
        meta_only: bool,
    ) -> RedisResult<AITensor<T, S>>
    where
        T: Debug,
        Vec<T>: ToFromBlob,
    {
        let args = modelget_cmd_build(key, meta_only);
        let tensor = if meta_only {
            if self.debug {
                println!("AI.TENSORGET {:?}", args)
            }
            let meta: AITensorMeta<T, S> = redis::cmd("AI.TENSORGET")
                .arg(args)
                .query_async(con)
                .await?;
            AITensor { meta, blob: vec![] }
        } else {
            let tensor: AITensor<T, S> = redis::cmd("AI.TENSORGET")
                .arg(args)
                .query_async(con)
                .await?;
            tensor
        };
        Ok(tensor)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::AIDataType;
    #[cfg(feature = "ndar")]
    use ndarray::{arr1, arr2, arr3};
    #[test]
    fn ai_tensor() {
        let tensor_data: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04];
        let shape: [usize; 1] = [4];
        let ai_tensor: AITensor<u8, 1> = AITensor::new(shape, tensor_data);

        let expected_tensor: AITensor<u8, 1> = AITensor {
            meta: AITensorMeta {
                dtype: AIDataType::UINT8,
                shape: [4],
                phantom: PhantomData,
            },
            blob: vec![0x01, 0x02, 0x03, 0x04],
        };

        assert_eq!(expected_tensor, ai_tensor)
    }
    #[test]
    fn one_dim_i8_ai_tensor() {
        let tensor_data: Vec<i8> = vec![0x01, 0x02, 0x03, 0x04];
        let shape: [usize; 1] = [4];
        let ai_tensor: AITensor<i8, 1> = AITensor::new(shape, tensor_data.to_blob());

        let expected_tensor: AITensor<i8, 1> = AITensor {
            meta: AITensorMeta {
                dtype: AIDataType::INT8,
                shape: [4],
                phantom: PhantomData,
            },
            blob: vec![0x01, 0x02, 0x03, 0x04],
        };

        assert_eq!(expected_tensor, ai_tensor)
    }
    #[test]
    fn one_dim_i32_ai_tensor() {
        let tensor_data: Vec<i32> = vec![1, 2, 3, 255];
        let shape: [usize; 1] = [4];
        let ai_tensor: AITensor<i32, 1> = AITensor::new(shape, tensor_data.to_blob());

        let expected_tensor: AITensor<i32, 1> = AITensor {
            meta: AITensorMeta {
                dtype: AIDataType::INT32,
                shape: [4],
                phantom: PhantomData,
            },
            blob: vec![0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 255],
        };

        assert_eq!(expected_tensor, ai_tensor)
    }
    #[test]
    fn one_dim_f64_ai_tensor() {
        let tensor_data: Vec<f64> = vec![1., 2., 3., 255.];
        let shape: [usize; 1] = [4];
        let ai_tensor: AITensor<f64, 1> = AITensor::new(shape, tensor_data.to_blob());

        let expected_tensor: AITensor<f64, 1> = AITensor {
            meta: AITensorMeta {
                dtype: AIDataType::DOUBLE,
                shape: [4],
                phantom: PhantomData,
            },
            blob: vec![
                63, 240, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 64, 8, 0, 0, 0, 0, 0, 0, 64,
                111, 224, 0, 0, 0, 0, 0,
            ],
        };

        assert_eq!(expected_tensor, ai_tensor)
    }
    #[test]
    #[cfg(feature = "ndar")]
    fn one_dim_ndarray_f32_ai_tensor() {
        let tensor: ndarray::Array1<f32> = arr1(&[1., 2., 3., 255.]);

        let ai_tensor: AITensor<f32, 1> = tensor.into();

        let expected_tensor: AITensor<f32, 1> = AITensor {
            meta: AITensorMeta {
                dtype: AIDataType::FLOAT,
                shape: [4],
                phantom: PhantomData,
            },
            blob: vec![63, 128, 0, 0, 64, 0, 0, 0, 64, 64, 0, 0, 67, 127, 0, 0],
        };

        assert_eq!(expected_tensor, ai_tensor)
    }
    #[test]
    #[cfg(feature = "ndar")]
    fn one_dim_ndarray_f64_ai_tensor() {
        let tensor_data: Vec<f64> = vec![1., 2., 3., 255.];
        let shape: [usize; 1] = [4];
        let ai_tensor: AITensor<f64, 1> = AITensor::new(shape, tensor_data.to_blob());

        let expected_tensor: AITensor<f64, 1> = AITensor {
            meta: AITensorMeta {
                dtype: AIDataType::DOUBLE,
                shape: [4],
                phantom: PhantomData,
            },
            blob: vec![
                63, 240, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 64, 8, 0, 0, 0, 0, 0, 0, 64,
                111, 224, 0, 0, 0, 0, 0,
            ],
        };

        assert_eq!(expected_tensor, ai_tensor)
    }
    #[test]
    #[cfg(feature = "ndar")]
    fn two_dim_ndarray_f32_ai_tensor() {
        let tensor: ndarray::Array2<f32> = arr2(&[[1., 2., 3., 255.], [1., 2., 3., 255.]]);

        let ai_tensor: AITensor<f32, 2> = tensor.into();

        let expected_tensor: AITensor<f32, 2> = AITensor {
            meta: AITensorMeta {
                dtype: AIDataType::FLOAT,
                shape: [2, 4],
                phantom: PhantomData,
            },
            blob: vec![
                63, 128, 0, 0, 64, 0, 0, 0, 64, 64, 0, 0, 67, 127, 0, 0, 63, 128, 0, 0, 64, 0, 0,
                0, 64, 64, 0, 0, 67, 127, 0, 0,
            ],
        };

        assert_eq!(expected_tensor, ai_tensor)
    }
    #[test]
    #[should_panic(expected = "impossible to convert a 2 ndarray into an 3 AITensor")]
    #[cfg(feature = "ndar")]
    fn two_dim_ndarray_f32_ai_tensor_go_wrong() {
        let tensor: ndarray::Array2<f32> = arr2(&[[1., 2., 3., 255.], [1., 2., 3., 255.]]);
        let _ai_tensor: AITensor<f32, 3> = tensor.into();
    }
    #[test]
    #[cfg(feature = "ndar")]
    fn three_dim_ndarray_f32_ai_tensor() {
        let tensor: ndarray::Array3<f32> = arr3(&[
            [[1., 2.], [3., 255.]],
            [[1., 2.], [3., 5.]],
            [[1., 2.], [3., 255.]],
        ]);

        let ai_tensor: AITensor<f32, 3> = tensor.into();

        let expected_tensor: AITensor<f32, 3> = AITensor {
            meta: AITensorMeta {
                dtype: AIDataType::FLOAT,
                shape: [3, 2, 2],
                phantom: PhantomData,
            },
            blob: vec![
                63, 128, 0, 0, 64, 0, 0, 0, 64, 64, 0, 0, 67, 127, 0, 0, 63, 128, 0, 0, 64, 0, 0,
                0, 64, 64, 0, 0, 64, 160, 0, 0, 63, 128, 0, 0, 64, 0, 0, 0, 64, 64, 0, 0, 67, 127,
                0, 0,
            ],
        };

        assert_eq!(expected_tensor, ai_tensor)
    }
}
