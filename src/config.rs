/// Documentation for the config interface
use crate::Backend;
use crate::RedisAIClient;
use redis::RedisResult;

// AI.CONFIG LOADBACKEND TORCH redisai_torch/redisai_torch.so
// AI.CONFIG LOADBACKEND TORCH /usr/lib/redis/modules/redisai/backends/redisai_torch/redisai_torch.so
// AI.CONFIG LOADBACKEND ONNX /usr/lib/redis/modules/backends/redisai_onnxruntime/redisai_onnxruntime.so
fn loadbackend_cmd_build(identifier: Backend, path: String) -> Vec<String> {
    vec![identifier.to_string(), path]
}

impl RedisAIClient {
    pub fn ai_loadbackend(
        &self,
        con: &mut redis::Connection,
        identifier: Backend,
        path: String,
    ) -> RedisResult<()> {
        let args = loadbackend_cmd_build(identifier, path);
        if self.debug {
            format!("AI.CONFIG LOADBACKEND {:?}", args);
        }
        redis::cmd("AI.CONFIG")
            .arg("LOADBACKEND")
            .arg(args)
            .query(con)?;
        Ok(())
    }
}

#[cfg(feature = "aio")]
impl RedisAIClient {
    pub async fn ai_loadbackend_async(
        &self,
        con: &mut redis::aio::Connection,
        identifier: Backend,
        path: String,
    ) -> RedisResult<()> {
        let args = loadbackend_cmd_build(identifier, path);
        if self.debug {
            format!("AI.CONFIG LOADBACKEND {:?}", args);
        }
        redis::cmd("AI.CONFIG")
            .arg("LOADBACKEND")
            .arg(args)
            .query_async(con)
            .await?;
        Ok(())
    }
}
