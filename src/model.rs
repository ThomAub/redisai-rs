/// Documentation for the model api
use crate::{Backend, Device, RedisAIClient};
use redis::RedisResult;

use std::fs::File;
use std::io::prelude::Read;
use std::path::Path;
/// The Model metadata
#[derive(Debug, PartialEq)]
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
#[derive(Debug, PartialEq, Default)]
pub struct AIModel {
    /// The Model metadata
    pub meta: AIModelMeta,
    /// The model blob representation
    pub blob: Vec<u8>,
}

impl AIModel {
    pub fn new_from_file(meta: AIModelMeta, path: &Path) -> RedisResult<AIModel> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::<u8>::new();
        file.read_to_end(&mut buffer).unwrap();
        Ok(AIModel { meta, blob: buffer })
    }
}

impl RedisAIClient {
    pub fn ai_modelset(
        &self,
        con: &mut redis::Connection,
        key: String,
        model: AIModel,
    ) -> RedisResult<()> {
        let backend_str = model.meta.backend.to_string();
        let device_str = model.meta.device.to_string();

        let mut debug_command = format!("AI.MODELSET {}", &key);
        let mut args_command: Vec<String> = vec![];

        if let Some(tag) = model.meta.tag {
            debug_command = debug_command + " TAG " + &tag;
            args_command.append(&mut vec!["TAG".to_string(), tag]);
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
                debug_command = debug_command + " BATCHSIZE " + &n.to_string();
                args_command.append(&mut vec!["BATCHSIZE".to_string(), n.to_string()]);
            }
            (n, m, 0) if m > n => {
                panic!(
                    "BATCHSIZE n={} should be greater than MINBATCHSIZE m={}",
                    n, m
                )
            }
            (n, m, 0) => {
                debug_command = debug_command + " BATCHSIZE " + &n.to_string();
                args_command.append(&mut vec!["BATCHSIZE".to_string(), n.to_string()]);

                debug_command = debug_command + " MINBATCHSIZE " + &m.to_string();
                args_command.append(&mut vec!["MINBATCHSIZE".to_string(), m.to_string()]);
            }
            (n, m, t) if m > n => {
                panic!(
                    "Even with MINBATCHTIMEOUT t={}, BATCHSIZE n={} should be greater than MINBATCHSIZE m={}",
                    t, n, m
                )
            }
            (n, m, t) => {
                debug_command = debug_command + " BATCHSIZE " + &n.to_string();
                args_command.append(&mut vec!["BATCHSIZE".to_string(), n.to_string()]);

                debug_command = debug_command + " MINBATCHSIZE " + &m.to_string();
                args_command.append(&mut vec!["MINBATCHSIZE".to_string(), m.to_string()]);

                debug_command = debug_command + " MINBATCHTIMEOUT " + &t.to_string();
                args_command.append(&mut vec!["MINBATCHTIMEOUT".to_string(), t.to_string()]);
            }
        }

        // Handling the inputs outputs only for only for TensorFlow models
        // TODO: check for Tflite ?
        match model.meta.backend {
            Backend::TF | Backend::TFLITE => {
                if let Some(inputs) = model.meta.inputs {
                    let inputs_joined = inputs.join(" ");
                    debug_command = debug_command + " INPUTS " + &inputs_joined;
                    args_command.append(&mut vec!["INPUTS".to_string(), inputs_joined]);
                } else {
                    panic!("Trying to use a TF or TFlite model without setting INPUTS tensors")
                }

                if let Some(outputs) = model.meta.outputs {
                    let outputs_joined = outputs.join(" ");
                    debug_command = debug_command + " INPUTS " + &outputs_joined;
                    args_command.append(&mut vec!["INPUTS".to_string(), outputs_joined]);
                } else {
                    panic!("Trying to use a TF or TFlite model without setting OUTPUTS tensors")
                }
            }
            _ => {}
        }

        if self.debug {
            println!("{} BLOB ", &debug_command); //{:#04X?} &model.blob
        }

        redis::cmd("AI.MODELSET")
            .arg(key)
            .arg(backend_str)
            .arg(device_str)
            .arg(args_command)
            .arg("BLOB")
            .arg(model.blob)
            .query(con)?;
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
}
