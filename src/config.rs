/// Documentation for the config interface
use crate::RedisAIClient;
use redis::RedisResult;

// AI.CONFIG LOADBACKEND TORCH redisai_torch/redisai_torch.so
// AI.CONFIG LOADBACKEND TORCH /usr/lib/redis/modules/redisai/backends/redisai_torch/redisai_torch.so
// AI.CONFIG LOADBACKEND ONNX /usr/lib/redis/modules/backends/redisai_onnxruntime/redisai_onnxruntime.so
impl RedisAIClient {
    pub fn ai_loadbackend(
        &self,
        con: &mut redis::Connection,
        identifier: String,
        path: String,
    ) -> RedisResult<()> {
        redis::cmd("AI.CONFIG")
            .arg("LOADBACKEND")
            .arg(identifier)
            .arg(path)
            .query(con)?;
        Ok(())
    }
}
