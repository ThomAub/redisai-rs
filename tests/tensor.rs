#[cfg(feature = "ndar")]
use ndarray::{arr2, arr3};

use redisai::tensor::{AITensor, ToFromBlob};
use redisai::RedisAIClient;
#[test]
fn ai_tensorset_one_dim_int8() {
    let aiclient: RedisAIClient = RedisAIClient { debug: true };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: Vec<i8> = vec![1, 2, 3, 127];
    let shape: [usize; 1] = [4];
    let ai_tensor: AITensor<i8, 1> = AITensor::new(shape, tensor_data.to_blob());
    assert_eq!(
        Ok(()),
        aiclient.ai_tensorset(&mut con, "one_dim_i8_tensor".to_string(), ai_tensor)
    );
}

#[test]
fn ai_tensorget_one_dim_int8() {
    let aiclient: RedisAIClient = RedisAIClient { debug: true };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: Vec<i8> = vec![1, 2, 3, 127];
    let shape: [usize; 1] = [4];
    let expected_ai_tensor: AITensor<i8, 1> = AITensor::new(shape, tensor_data.to_blob());
    let key = "one_dim_i8_tensor".to_string();
    let _ = aiclient.ai_tensorset(&mut con, key.clone(), expected_ai_tensor.clone());

    let ai_tensor: AITensor<i8, 1> = aiclient.ai_tensorget(&mut con, key, false).unwrap();

    assert_eq!(expected_ai_tensor, ai_tensor);
}
#[test]
fn ai_tensorget_one_dim_int8_meta_only() {
    let aiclient: RedisAIClient = RedisAIClient { debug: true };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: Vec<i8> = vec![1, 2, 3, 127];
    let shape: [usize; 1] = [4];
    let mut expected_ai_tensor: AITensor<i8, 1> = AITensor::new(shape, tensor_data.to_blob());

    let key = "one_dim_i8_tensor_get_meta".to_string();
    let _ = aiclient
        .ai_tensorset(&mut con, key.clone(), expected_ai_tensor.clone())
        .unwrap();

    let ai_tensor: AITensor<i8, 1> = aiclient.ai_tensorget(&mut con, key, true).unwrap();
    expected_ai_tensor.blob = vec![];
    assert_eq!(expected_ai_tensor, ai_tensor);
}
#[test]
#[should_panic(
    expected = "value: Size of the retrieve data shape doesn't match the expected const shape"
)]
fn ai_tensorget_wrong_shape() {
    let aiclient: RedisAIClient = RedisAIClient { debug: true };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: Vec<i8> = vec![1, 2, 3, 4];
    let shape: [usize; 1] = [4];
    let expected_ai_tensor: AITensor<i8, 1> = AITensor::new(shape, tensor_data.to_blob());

    let key = "one_dim_i8_tensor_get_wrong".to_string();
    let _ = aiclient.ai_tensorset(&mut con, key.clone(), expected_ai_tensor.clone());

    let _: AITensor<i8, 2> = aiclient.ai_tensorget(&mut con, key, true).unwrap();
}
#[test]
fn ai_tensorset_three_dim_int32() {
    let aiclient: RedisAIClient = RedisAIClient { debug: false };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let shape: [usize; 3] = [2, 2, 3];
    let ai_tensor: AITensor<i32, 3> = AITensor::new(shape, tensor_data.to_blob());

    assert_eq!(
        Ok(()),
        aiclient.ai_tensorset(&mut con, "three_dim_i32_tensor".to_string(), ai_tensor)
    );
}
#[test]
fn ai_tensorget_three_dim_int32() {
    let aiclient: RedisAIClient = RedisAIClient { debug: false };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let shape: [usize; 3] = [2, 2, 3];
    let expected_ai_tensor: AITensor<i32, 3> = AITensor::new(shape, tensor_data.to_blob());

    let key = "three_dim_i32_tensor_get".to_string();
    let _ = aiclient.ai_tensorset(&mut con, key.clone(), expected_ai_tensor.clone());

    assert_eq!(
        Ok(expected_ai_tensor),
        aiclient.ai_tensorget(&mut con, key, false)
    );
}
#[test]
fn ai_tensorget_three_dim_int32_meta_only() {
    let aiclient: RedisAIClient = RedisAIClient { debug: false };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let shape: [usize; 3] = [2, 2, 3];
    let mut expected_ai_tensor: AITensor<i32, 3> = AITensor::new(shape, tensor_data.to_blob());

    let key = "three_dim_i32_tensor_meta".to_string();
    let _ = aiclient
        .ai_tensorset(&mut con, key.clone(), expected_ai_tensor.clone())
        .unwrap();

    expected_ai_tensor.blob = vec![];
    assert_eq!(
        Ok(expected_ai_tensor),
        aiclient.ai_tensorget(&mut con, key, true)
    );
}
#[test]
fn ai_tensorset_one_dim_float32() {
    let aiclient: RedisAIClient = RedisAIClient { debug: true };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: Vec<f32> = vec![1., 2., 3., 4.];
    let shape: [usize; 1] = [4];
    let ai_tensor: AITensor<f32, 1> = AITensor::new(shape, tensor_data.to_blob());

    assert_eq!(
        Ok(()),
        aiclient.ai_tensorset(&mut con, "one_dim_f32_tensor".to_string(), ai_tensor)
    );
}
#[test]
fn ai_tensorget_one_dim_float32() {
    let aiclient: RedisAIClient = RedisAIClient { debug: false };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: Vec<f32> = vec![1., 2., 3., 4.];
    let shape: [usize; 1] = [4];
    let expected_ai_tensor: AITensor<f32, 1> = AITensor::new(shape, tensor_data.to_blob());

    let key = "one_dim_f32_tensor_get".to_string();
    let _ = aiclient.ai_tensorset(&mut con, key.clone(), expected_ai_tensor.clone());

    assert_eq!(
        Ok(expected_ai_tensor),
        aiclient.ai_tensorget(&mut con, key, false)
    );
}
#[test]
fn ai_tensorset_three_dim_float32() {
    let aiclient: RedisAIClient = RedisAIClient { debug: false };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let shape: [usize; 3] = [2, 2, 3];
    let ai_tensor: AITensor<f32, 3> = AITensor::new(shape, tensor_data.to_blob());

    assert_eq!(
        Ok(()),
        aiclient.ai_tensorset(&mut con, "three_dim_f32_tensor".to_string(), ai_tensor)
    );
}
#[test]
fn ai_tensorget_three_dim_float32() {
    let aiclient: RedisAIClient = RedisAIClient { debug: false };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let shape: [usize; 3] = [2, 2, 3];
    let expected_ai_tensor: AITensor<f32, 3> = AITensor::new(shape, tensor_data.to_blob());

    let key = "three_dim_f32_tensor_get".to_string();
    let _ = aiclient.ai_tensorset(&mut con, key.clone(), expected_ai_tensor.clone());

    assert_eq!(
        Ok(expected_ai_tensor),
        aiclient.ai_tensorget(&mut con, key, false)
    );
}

#[test]
fn ai_tensorset_one_dim_float64() {
    let aiclient: RedisAIClient = RedisAIClient { debug: false };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: Vec<f64> = vec![1., 2., 3., 4.];
    let shape: [usize; 1] = [4];
    let ai_tensor: AITensor<f64, 1> = AITensor::new(shape, tensor_data.to_blob());

    assert_eq!(
        Ok(()),
        aiclient.ai_tensorset(&mut con, "one_dim_double_tensor".to_string(), ai_tensor)
    );
}
#[test]
fn ai_tensorget_one_dim_float64() {
    let aiclient: RedisAIClient = RedisAIClient { debug: true };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: Vec<f64> = vec![1., 2., 3., 4.];
    let shape: [usize; 1] = [4];
    let expected_ai_tensor: AITensor<f64, 1> = AITensor::new(shape, tensor_data.to_blob());

    let key = "one_dim_double_tensor_get".to_string();
    let _ = aiclient.ai_tensorset(&mut con, key.clone(), expected_ai_tensor.clone());

    assert_eq!(
        Ok(expected_ai_tensor),
        aiclient.ai_tensorget(&mut con, key, false)
    );
}
#[test]
#[cfg(feature = "ndar")]
fn ai_tensorset_from_2d_ndarray() {
    let aiclient: RedisAIClient = RedisAIClient { debug: false };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor_data: ndarray::Array2<f64> = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
    let shape: [usize; 2] = [2, 3];
    let ai_tensor: AITensor<f64, 2> = AITensor::new(shape, tensor_data.into_raw_vec().to_blob());

    assert_eq!(
        Ok(()),
        aiclient.ai_tensorset(
            &mut con,
            "two_dim_double_ndarray_tensor".to_string(),
            ai_tensor
        )
    );
}
#[test]
#[cfg(feature = "ndar")]
fn ai_tensorget_from_2d_ndarray() {
    let aiclient: RedisAIClient = RedisAIClient { debug: false };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor: ndarray::Array2<f64> = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
    let expected_ai_tensor: AITensor<f64, 2> = tensor.into(); //AITensor::new(shape, tensor_data.into_raw_vec().to_blob());

    let key = "two_dim_double_ndarray_tensor_get".to_string();
    let _ = aiclient.ai_tensorset(&mut con, key.clone(), expected_ai_tensor.clone());

    assert_eq!(
        Ok(expected_ai_tensor),
        aiclient.ai_tensorget(&mut con, key, false)
    );
}
#[test]
#[cfg(feature = "ndar")]
fn ai_tensorset_from_3d_ndarray() {
    let aiclient: RedisAIClient = RedisAIClient { debug: true };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor: ndarray::Array3<f32> = arr3(&[
        [[1., 2.], [3., 4.]],
        [[5., 6.], [7., 8.]],
        [[9., 0.], [1., 2.]],
    ]);
    let ai_tensor: AITensor<f32, 3> = tensor.into(); //AITensor::new(shape, tensor_data.into_raw_vec().to_blob());
    assert_eq!(
        Ok(()),
        aiclient.ai_tensorset(
            &mut con,
            "three_dim_float_ndarray_tensor".to_string(),
            ai_tensor
        )
    );
}
#[test]
#[cfg(feature = "ndar")]
fn ai_tensorget_from_3d_ndarray() {
    let aiclient: RedisAIClient = RedisAIClient { debug: false };
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let tensor: ndarray::Array3<f32> = arr3(&[
        [[1., 2.], [3., 4.]],
        [[5., 6.], [7., 8.]],
        [[9., 0.], [1., 2.]],
    ]);
    let expected_ai_tensor: AITensor<f32, 3> = tensor.into();

    let key = "three_dim_float_ndarray_tensor_get".to_string();
    let _ = aiclient.ai_tensorset(&mut con, key.clone(), expected_ai_tensor.clone());

    assert_eq!(
        Ok(expected_ai_tensor),
        aiclient.ai_tensorget(&mut con, key, false)
    );
}
