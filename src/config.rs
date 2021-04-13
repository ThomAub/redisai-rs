/// Documentation for the config interface
use crate::RedisAIClient;
use redis::RedisResult;

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
