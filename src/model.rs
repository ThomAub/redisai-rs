/// Documentation for the model api
use crate::{Backend, Device, RedisAIClient};
use redis::{FromRedisValue, RedisResult, Value};

use std::io::prelude::Read;
use std::path::Path;
use std::str::FromStr;

#[cfg(feature = "tokio-comp")]
use tokio::io::AsyncReadExt;

#[cfg(feature = "async-std-comp")]
use async_std::prelude::*;

/// The Model metadata
#[derive(Debug, Clone, PartialEq)]
pub struct AIModelMeta {
    pub backend: Backend,
    /// The type of device that will execute the model
    pub device: Device,
    /// An optional string for tagging the model such as a version number or any arbitrary identifier
    pub tag: Option<String>,
    /// When provided with an batchsize n that is greater than 0,
    /// the engine will batch incoming requests from multiple clients
    /// that use the model with input tensors of the same shape.
    /// When AI.MODELRUN is called the requests queue is visited and input tensors
    /// from compatible requests are concatenated along the 0th (batch) dimension until n is exceeded.
    /// The model is then run for the entire batch and the results are unpacked back to the individual requests unblocking
    /// their respective clients. If the batch size of the inputs to of first request in
    /// the queue exceeds BATCHSIZE , the request is served immediately (default value: 0).
    pub batchsize: isize,
    /// When provided with an m that is greater than 0,
    /// the engine will postpone calls to AI.MODELRUN until the batch's size had reached m.
    /// In this case, note that requests for which m is not reached will hang indefinitely (default value: 0),
    /// unless MINBATCHTIMEOUT is provided.
    pub min_batchsize: isize,
    /// When provided with a t (expressed in milliseconds) that is greater than 0,
    /// the engine will trigger a run even though MINBATCHSIZE has not been reached after t milliseconds
    /// from the time a MODELRUN (or the enclosing DAGRUN ) is enqueued.
    /// This only applies to cases where both BATCHSIZE and MINBATCHSIZE are greater than 0.
    pub min_batch_timeout: isize,
    /// One or more names of the model's input nodes (applicable only for TensorFlow models)
    pub inputs: Option<Vec<String>>,
    /// One or more names of the model's output nodes (applicable only for TensorFlow models)
    pub outputs: Option<Vec<String>>,
}

impl Default for AIModelMeta {
    fn default() -> Self {
        AIModelMeta {
            backend: Backend::ONNX,
            device: Device::CPU,
            tag: None,
            batchsize: 0,
            min_batchsize: 0,
            min_batch_timeout: 0,
            inputs: None,
            outputs: None,
        }
    }
}
/// The Model represents a computation graph by one of the supported DL/ML framework backends
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AIModel {
    /// The Model metadata
    pub meta: AIModelMeta,
    /// The model blob representation
    pub blob: Vec<u8>,
}

impl AIModel {
    /// Read a model file from a path
    pub fn new_from_file(meta: AIModelMeta, path: &Path) -> RedisResult<AIModel> {
        let mut file = std::fs::File::open(path)?;
        let mut buffer = Vec::<u8>::new();
        file.read_to_end(&mut buffer).unwrap();
        Ok(AIModel { meta, blob: buffer })
    }
    /// Async Read a model file from a path
    #[cfg(feature = "tokio-comp")]
    pub async fn new_from_file_tokio(meta: AIModelMeta, path: &Path) -> RedisResult<AIModel> {
        let mut file = tokio::fs::File::open(path).await?;
        let mut buffer = Vec::<u8>::new();
        file.read_to_end(&mut buffer).await?;
        Ok(AIModel { meta, blob: buffer })
    }
    /// Async Read a model file from a path
    #[cfg(feature = "async-std-comp")]
    pub async fn new_from_file_async_std(meta: AIModelMeta, path: &Path) -> RedisResult<AIModel> {
        let mut file = async_std::fs::File::open(path).await?;
        let mut buffer = Vec::<u8>::new();
        file.read_to_end(&mut buffer).await?;
        Ok(AIModel { meta, blob: buffer })
    }
}
impl FromRedisValue for AIModelMeta {
    fn from_redis_value(v: &Value) -> RedisResult<Self> {
        let mut it = v.as_sequence().unwrap().iter();
        let meta: AIModelMeta = match (
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
        ) {
            (
                _, //Some(backend_field),
                Some(backend),
                _, //Some(device_field),
                Some(device),
                _, //Some(tag_field),
                Some(tag),
                _, //Some(batchsize_field),
                Some(batchsize),
                _, //Some(minbatchsize_field),
                Some(minbatchsize),
                _, //Some(inputs_field),
                Some(inputs),
                _, //Some(outputs_field),
                Some(outputs),
            ) => {
                let backend = Backend::from_str(&String::from_redis_value(backend)?).unwrap();
                let (inputs, outputs) = match backend {
                    Backend::TF => (
                        Vec::<String>::from_redis_value(inputs).ok(),
                        Vec::<String>::from_redis_value(outputs).ok(),
                    ),
                    _ => (None, None),
                };
                let mut tag = String::from_redis_value(tag).ok();
                if tag == Some("".to_string()) {
                    tag = None
                }

                AIModelMeta {
                    backend,
                    device: Device::from_str(&String::from_redis_value(device)?).unwrap(),
                    tag,
                    batchsize: isize::from_redis_value(batchsize)?,
                    min_batchsize: isize::from_redis_value(minbatchsize)?,
                    min_batch_timeout: 0, //TODO: NOT in META ?
                    inputs: inputs,
                    outputs: outputs,
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
impl FromRedisValue for AIModel {
    fn from_redis_value(v: &Value) -> RedisResult<Self> {
        let it = v.as_sequence().unwrap().iter();
        let meta_part_iter = it.clone().take(4);
        let meta_part = Value::Bulk(
            meta_part_iter
                .map(|elem| elem.clone())
                .collect::<Vec<Value>>(),
        );
        let meta: AIModelMeta = AIModelMeta::from_redis_value(&meta_part).unwrap();

        let mut blob_part_iter = it.skip(4);
        let blob = match (blob_part_iter.next(), blob_part_iter.next()) {
            (Some(_blob_field), Some(blob)) => {
                let blob_bytes = Vec::<u8>::from_redis_value(blob).unwrap();
                blob_bytes
            }
            _ => {
                return Err(redis::RedisError::from((
                    redis::ErrorKind::TypeError,
                    "Cannot convert AIModel blob field to some string",
                )))
            }
        };
        Ok(Self { meta, blob })
    }
}

fn modelset_cmd_build(key: String, model: &AIModel) -> Vec<String> {
    let mut args_command: Vec<String> = vec![key];
    args_command.push(model.meta.backend.to_string());
    args_command.push(model.meta.device.to_string());

    if let Some(tag) = &model.meta.tag {
        args_command.append(&mut vec!["TAG".to_string(), tag.to_string()]);
    }
    // Handling the batchsize and the min_batchsize and the min_batch_timeout
    match (
        model.meta.batchsize,
        model.meta.min_batchsize,
        model.meta.min_batch_timeout,
    ) {
        (0, 0, 0) => {}
        (0, 0, t) => {
            panic!("MINBATCHTIMEOUT t={} should only be set where both BATCHSIZE and MINBATCHSIZE are greater than 0.", t)
        }
        (n, 0, 0) => {
            args_command.append(&mut vec!["BATCHSIZE".to_string(), n.to_string()]);
        }
        (n, m, 0) if m > n => {
            panic!(
                "BATCHSIZE n={} should be greater than MINBATCHSIZE m={}",
                n, m
            )
        }
        (n, m, 0) => {
            args_command.append(&mut vec!["BATCHSIZE".to_string(), n.to_string()]);
            args_command.append(&mut vec!["MINBATCHSIZE".to_string(), m.to_string()]);
        }
        (n, m, t) if m > n => {
            panic!(
                    "Even with MINBATCHTIMEOUT t={}, BATCHSIZE n={} should be greater than MINBATCHSIZE m={}",
                    t, n, m
                )
        }
        (n, m, t) => {
            args_command.append(&mut vec!["BATCHSIZE".to_string(), n.to_string()]);
            args_command.append(&mut vec!["MINBATCHSIZE".to_string(), m.to_string()]);
            args_command.append(&mut vec!["MINBATCHTIMEOUT".to_string(), t.to_string()]);
        }
    }

    // Handling the inputs outputs only for only for TensorFlow models
    // TODO: check for Tflite ?
    match model.meta.backend {
        Backend::TF | Backend::TFLITE => {
            if let Some(inputs) = &model.meta.inputs {
                //let inputs_joined = inputs.join(" ");
                args_command.push("INPUTS".to_string()); //, inputs_joined]);
                args_command.append(&mut inputs.to_owned());
            } else {
                panic!("Trying to use a TF or TFlite model without setting INPUTS tensors")
            }

            if let Some(outputs) = &model.meta.outputs {
                args_command.push("OUTPUTS".to_string()); //, inputs_joined]);
                args_command.append(&mut outputs.to_owned());
            } else {
                panic!("Trying to use a TF or TFlite model without setting OUTPUTS tensors")
            }
        }
        _ => {}
    }
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
fn modelrun_cmd_build(
    key: String,
    timeout: i64,
    inputs: Vec<String>,
    outputs: Vec<String>,
) -> Vec<String> {
    let mut args_command: Vec<String> = vec![key];
    args_command.push("TIMEOUT".to_string());
    args_command.push(timeout.to_string());
    args_command.push("INPUTS".to_string());
    args_command.append(&mut inputs.clone());
    args_command.push("OUTPUTS".to_string());
    args_command.append(&mut outputs.clone());
    args_command
}
impl RedisAIClient {
    pub fn ai_modelset(
        &self,
        con: &mut redis::Connection,
        key: String,
        model: AIModel,
    ) -> RedisResult<()> {
        let args = modelset_cmd_build(key, &model);
        if self.debug {
            format!("AI.MODELSTORE {:?}", &args);
        }
        redis::cmd("AI.MODELSTORE")
            .arg(args)
            .arg("BLOB")
            .arg(model.blob)
            .query(con)?;
        Ok(())
    }
    pub fn ai_modelget(
        &self,
        con: &mut redis::Connection,
        key: String,
        meta_only: bool,
    ) -> RedisResult<AIModel> {
        let args = modelget_cmd_build(key, meta_only);
        let model = if meta_only {
            if self.debug {
                println!("AI.MODELGET {:?}", args)
            }
            let meta: AIModelMeta = redis::cmd("AI.MODELGET").arg(args).query(con)?;
            AIModel { meta, blob: vec![] }
        } else {
            let model: AIModel = redis::cmd("AI.MODELGET").arg(args).query(con)?;
            model
        };
        Ok(model)
    }
    pub fn ai_modelscan(&self, con: &mut redis::Connection) -> RedisResult<Vec<Vec<String>>> {
        if self.debug {
            format!("AI._MODELSCAN");
        }
        let models: Vec<Vec<String>> = redis::cmd("AI._MODELSCAN").query(con)?;
        Ok(models)
    }
    pub fn ai_modeldel(&self, con: &mut redis::Connection, key: String) -> RedisResult<()> {
        if self.debug {
            format!("AI.MODELDEL {}", key);
        }
        redis::cmd("AI.MODELDEL").arg(key).query(con)?;
        Ok(())
    }
    pub fn ai_modelrun(
        &self,
        con: &mut redis::Connection,
        key: String,
        timeout: i64,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> RedisResult<()> {
        let args_command = modelrun_cmd_build(key, timeout, inputs, outputs);
        dbg!(&args_command);
        if self.debug {
            format!("AI.MODELRUN {:?}", args_command);
        }
        redis::cmd("AI.MODELRUN").arg(args_command).query(con)?;
        Ok(())
    }
}
#[cfg(feature = "aio")]
impl RedisAIClient {
    pub async fn ai_modelset_async(
        &self,
        con: &mut redis::aio::Connection,
        key: String,
        model: AIModel,
    ) -> RedisResult<()> {
        let args = modelset_cmd_build(key, &model);
        if self.debug {
            format!("AI.MODELSET {:?}", &args);
        }
        redis::cmd("AI.MODELSET")
            .arg(args)
            .arg("BLOB")
            .arg(model.blob)
            .query_async(con)
            .await?;
        Ok(())
    }
    pub async fn ai_modelscan_async(
        &self,
        con: &mut redis::aio::Connection,
    ) -> RedisResult<Vec<Vec<String>>> {
        if self.debug {
            format!("AI._MODELSCAN");
        }
        let models: Vec<Vec<String>> = redis::cmd("AI._MODELSCAN").query_async(con).await?;
        Ok(models)
    }
    pub async fn ai_modeldel_async(
        &self,
        con: &mut redis::aio::Connection,
        key: String,
    ) -> RedisResult<()> {
        if self.debug {
            format!("AI.MODELDEL {}", key);
        }
        redis::cmd("AI.MODELDEL").arg(key).query_async(con).await?;
        Ok(())
    }
    pub async fn ai_modelrun_async(
        &self,
        con: &mut redis::aio::Connection,
        key: String,
        timeout: i64,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> RedisResult<()> {
        let args_command = modelrun_cmd_build(key, timeout, inputs, outputs);
        if self.debug {
            format!("AI.MODELRUN {:?}", args_command);
        }
        redis::cmd("AI.MODELRUN")
            .arg(args_command)
            .query_async(con)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, RedisAIClient};

    #[test]
    fn ai_model_onnx_default() {
        let aiclient: RedisAIClient = RedisAIClient { debug: true };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();

        let ai_modelmeta = AIModelMeta {
            backend: Backend::ONNX,
            ..Default::default()
        };
        let model_path = Path::new("tests/testdata/findsquare.onnx");
        let key = "model:findsquare:onnx:".to_string();

        let ai_model = AIModel::new_from_file(ai_modelmeta, &model_path).unwrap();

        aiclient.ai_modelset(&mut con, key, ai_model).unwrap();
    }
    #[test]
    fn ai_model_onnx_default_with_tag() {
        let aiclient: RedisAIClient = RedisAIClient { debug: true };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();

        let ai_modelmeta = AIModelMeta {
            backend: Backend::ONNX,
            tag: Some("V1.3".to_string()),
            ..Default::default()
        };
        let model_path = Path::new("tests/testdata/findsquare.onnx");
        let key = "model:findsquare:onnx:tag".to_string();

        let ai_model = AIModel::new_from_file(ai_modelmeta, &model_path).unwrap();

        aiclient.ai_modelset(&mut con, key, ai_model).unwrap();
    }
    #[test]
    fn ai_model_onnx_default_with_batchsize() {
        let aiclient: RedisAIClient = RedisAIClient { debug: true };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();

        let ai_modelmeta = AIModelMeta {
            backend: Backend::ONNX,
            batchsize: 3,
            ..Default::default()
        };
        let model_path = Path::new("tests/testdata/findsquare.onnx");
        let key = "model:findsquare:onnx:batchsize:3".to_string();

        let ai_model = AIModel::new_from_file(ai_modelmeta, &model_path).unwrap();

        aiclient.ai_modelset(&mut con, key, ai_model).unwrap();
    }
    #[test]
    fn ai_model_onnx_default_with_batchsize_minbatchsize() {
        let aiclient: RedisAIClient = RedisAIClient { debug: true };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();

        let ai_modelmeta = AIModelMeta {
            backend: Backend::ONNX,
            batchsize: 3,
            min_batchsize: 2,
            ..Default::default()
        };
        let model_path = Path::new("tests/testdata/findsquare.onnx");
        let key = "model:findsquare:onnx:batchsize:3min_batchsize:2".to_string();

        let ai_model = AIModel::new_from_file(ai_modelmeta, &model_path).unwrap();

        aiclient.ai_modelset(&mut con, key, ai_model).unwrap();
    }
    #[test]
    #[should_panic(expected = "BATCHSIZE n=3 should be greater than MINBATCHSIZE m=5")]
    fn ai_model_onnx_default_with_batchsize_wrong_minbatchsize() {
        let aiclient: RedisAIClient = RedisAIClient { debug: true };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();

        let ai_modelmeta = AIModelMeta {
            backend: Backend::ONNX,
            batchsize: 3,
            min_batchsize: 5,
            ..Default::default()
        };
        let model_path = Path::new("tests/testdata/findsquare.onnx");
        let key = "model:findsquare:onnx:batchsize:3min_batchsize:2".to_string();

        let ai_model = AIModel::new_from_file(ai_modelmeta, &model_path).unwrap();

        aiclient.ai_modelset(&mut con, key, ai_model).unwrap();
    }
    #[test]
    #[should_panic(
        expected = "MINBATCHTIMEOUT t=5 should only be set where both BATCHSIZE and MINBATCHSIZE are greater than 0"
    )]
    fn ai_model_onnx_default_with_batchsize_wrong_min_batch_timeout() {
        let aiclient: RedisAIClient = RedisAIClient { debug: true };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();

        let ai_modelmeta = AIModelMeta {
            backend: Backend::ONNX,
            batchsize: 0,
            min_batchsize: 0,
            min_batch_timeout: 5,
            ..Default::default()
        };
        let model_path = Path::new("tests/testdata/findsquare.onnx");
        let key = "model:findsquare:onnx:batchsize:3min_batchsize:2".to_string();

        let ai_model = AIModel::new_from_file(ai_modelmeta, &model_path).unwrap();

        aiclient.ai_modelset(&mut con, key, ai_model).unwrap();
    }
    #[test]
    fn ai_model_scan() {
        let aiclient: RedisAIClient = RedisAIClient { debug: true };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();

        let model_path = Path::new("tests/testdata/findsquare.onnx");

        let ai_modelmeta_1 = AIModelMeta::default();
        let key_1 = "model:scan:1".to_string();

        let ai_model_1 = AIModel::new_from_file(ai_modelmeta_1, &model_path).unwrap();
        aiclient.ai_modelset(&mut con, key_1, ai_model_1).unwrap();

        let ai_modelmeta = AIModelMeta {
            tag: Some("V100".to_string()),
            ..Default::default()
        };
        let ai_model_1 = AIModel::new_from_file(ai_modelmeta, &model_path).unwrap();
        let key_2 = "model:scan:2".to_string();

        aiclient.ai_modelset(&mut con, key_2, ai_model_1).unwrap();

        let _models = aiclient.ai_modelscan(&mut con).unwrap();
        // assert_eq!(models, vec!["model:scan:1", "", "model:scan:2", "v100"])
    }
    #[test]
    fn ai_model_get() {
        let aiclient: RedisAIClient = RedisAIClient { debug: true };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();

        let model_path = Path::new("tests/testdata/findsquare.onnx");
        let ai_modelmeta_1 = AIModelMeta::default();
        let key_1 = "model:get:1".to_string();
        let ai_model_1 = AIModel::new_from_file(ai_modelmeta_1.clone(), &model_path).unwrap();
        aiclient
            .ai_modelset(&mut con, key_1.clone(), ai_model_1.clone())
            .unwrap();
        let model = aiclient.ai_modelget(&mut con, key_1.clone(), true).unwrap();
        dbg!(&model);
        let ai_model_expected = AIModel {
            meta: ai_modelmeta_1.clone(),
            blob: vec![],
        };
        assert_eq!(model, ai_model_expected)
    }
    #[test]
    fn ai_model_run() {
        use crate::tensor::{AITensor, ToFromBlob};

        let aiclient: RedisAIClient = RedisAIClient { debug: true };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();

        let model_path = Path::new("tests/testdata/graph.pb");
        //let model_path = Path::new("tests/testdata/pt-minimal.pt");

        let ai_modelmeta = AIModelMeta {
            inputs: Some(vec!["a".to_string(), "b".to_string()]),
            outputs: Some(vec!["mul".to_string()]),
            backend: Backend::TF,
            tag: Some("v1.0".to_string()),
            ..Default::default()
        };
        let key_1 = "model:tf:run:1".to_string();

        let ai_model = AIModel::new_from_file(ai_modelmeta, &model_path).unwrap();
        aiclient
            .ai_modelset(&mut con, key_1.clone(), ai_model)
            .unwrap();

        let input_a_key = "a".to_string();
        let tensor_data: Vec<f32> = vec![7., 3.];
        let shape: [usize; 1] = [2];
        let a_tensor: AITensor<f32, 1> = AITensor::new(shape, tensor_data.to_blob());

        let input_b_key = "b".to_string();
        let tensor_data: Vec<f32> = vec![3., 7.];
        let shape: [usize; 1] = [2];
        let b_tensor: AITensor<f32, 1> = AITensor::new(shape, tensor_data.to_blob());

        aiclient
            .ai_tensorset(&mut con, input_a_key.clone(), a_tensor)
            .unwrap();
        aiclient
            .ai_tensorset(&mut con, input_b_key.clone(), b_tensor)
            .unwrap();

        aiclient
            .ai_modelrun(
                &mut con,
                key_1.clone(),
                20000000,
                vec!["a".to_string(), "b".to_string()],
                vec!["mul".to_string()],
            )
            .unwrap();

        let ai_tensor: AITensor<f32, 1> = aiclient
            .ai_tensorget(&mut con, "mul".to_string(), false)
            .unwrap();

        let tensor_data: Vec<f32> = vec![21., 21.];
        let shape: [usize; 1] = [2];
        let expected_ai_tensor: AITensor<f32, 1> = AITensor::new(shape, tensor_data.to_blob());

        assert_eq!(expected_ai_tensor, ai_tensor)
    }
}
