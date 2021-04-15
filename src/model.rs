/// Documentation for the model api
use crate::{Backend, Device, RedisAIClient};
use redis::RedisResult;
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
    pub outputs: Option<String>,
}

impl AIModelMeta {
    pub fn new(
        backend: Backend,
        device: Device,
        tag: Option<String>,
        batchsize: isize,
        min_batchsize: isize,
        min_batch_timeout: isize,
        inputs: Option<Vec<String>>,
        outputs: Option<String>,
    ) -> Self {
        Self {
            backend,
            device,
            tag,
            batchsize,
            min_batchsize,
            min_batch_timeout,
            inputs,
            outputs,
        }
    }
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

#[derive(Debug, PartialEq, Default)]
pub struct AIModel {
    pub meta: AIModelMeta,
    pub blob: Vec<u8>,
}

impl RedisAIClient {
    pub fn ai_modelset(
        &self,
        con: &mut redis::Connection,
        key: String,
        backend: Backend,
        device: Device,
        model: Vec<u8>,
        tag: Option<String>,
        batchsize: Option<isize>,
        min_batchsize: Option<isize>,
        min_batch_timeout: Option<isize>,
        inputs: Option<String>,
        outputs: Option<String>,
    ) -> RedisResult<()> {
        let backend_str = backend.to_string();
        let device_str = device.to_string();

        if self.debug {
            println!(
                "AI.MODELSET {} {} {} BLOB {:#04X?}",
                &key, &backend_str, &device_str, model
            );
        }

        redis::cmd("AI.MODELSET")
            .arg(key)
            .arg(backend_str)
            .arg(device_str)
            .arg("BLOB")
            .arg(model)
            .query(con)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, Device, RedisAIClient};
    use std::fs::File;
    use std::io::prelude::Read;

    #[test]
    fn ai_model_TF_default() {
        let ai_modelmeta = AIModelMeta {
            backend: Backend::TF,
            ..Default::default()
        };
        let key = "HAHA";
        let meta = false;
        let mut command = format!("AI.TENSORGET {} MET", &key);
        if meta {
        } else {
            command = command + "BLOB"; // ).to_string();
        };
        dbg!(&command);
        // let ai_model = AIModel {
        //     meta: ai_modelmeta,
        //     ..Default::default()
        // };
        // dbg!(ai_model);
    }
    #[test]
    fn ai_modelset_() {
        let aiclient: RedisAIClient = RedisAIClient { debug: false };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();

        let mut file = File::open("tests/testdata/pt-minimal.pt").unwrap();
        let mut buffer = Vec::<u8>::new();
        file.read_to_end(&mut buffer).unwrap();

        //println!("{:?}", buffer);
        let key = "minimal_torch".to_string();
        let backend = Backend::TORCH;
        let device = Device::CPU;
        let model = buffer;
        let tag = None;
        let batchsize = None;
        let min_batchsize = None;
        let min_batch_timeout = None;
        let inputs = None;
        let outputs = None;
        aiclient
            .ai_modelset(
                &mut con,
                key,
                backend,
                device,
                model,
                tag,
                batchsize,
                min_batchsize,
                min_batch_timeout,
                inputs,
                outputs,
            )
            .unwrap();
        assert_eq!((), ())
    }
}
