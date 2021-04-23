/// Documentation for the info interface
use crate::RedisAIClient;
use redis::FromRedisValue;
use redis::{RedisResult, Value};

#[derive(Debug, PartialEq)]
pub struct AIInfo {
    version: String,
    low_level_api_version: String,
    rdb_encoding_version: String,
}
impl FromRedisValue for AIInfo {
    fn from_redis_value(v: &Value) -> RedisResult<Self> {
        let mut it = v.as_sequence().unwrap().iter();
        let (version, low_level_api_version, rdb_encoding_version): (String, String, String) =
            match (
                it.next(),
                it.next(),
                it.next(),
                it.next(),
                it.next(),
                it.next(),
            ) {
                (
                    _, //Some(version_field),
                    Some(version),
                    _, //Some(low_level_api_field),
                    Some(low_level_api_version),
                    _, //Some(rdb_encoding_field),
                    Some(rdb_encoding_version),
                ) => (
                    String::from_redis_value(version)?,
                    String::from_redis_value(low_level_api_version)?,
                    String::from_redis_value(rdb_encoding_version)?,
                ),
                _ => {
                    return Err(redis::RedisError::from((
                        redis::ErrorKind::TypeError,
                        "Cannot convert field to some string",
                    )))
                }
            };

        Ok(Self {
            version,
            low_level_api_version,
            rdb_encoding_version,
        })
    }
}
fn infoget_cmd_build(key: Option<String>) -> Vec<String> {
    let mut args_command: Vec<String> = vec![];
    if let Some(key) = key {
        args_command.push(key);
    }
    args_command
}
impl RedisAIClient {
    pub fn ai_infoget(
        &self,
        con: &mut redis::Connection,
        key: Option<String>,
    ) -> RedisResult<AIInfo> {
        let args = infoget_cmd_build(key);
        if self.debug {
            println!("RedisAI module running: AI.INFO {:?}", args)
        }
        let info: AIInfo = redis::cmd("AI.INFO").arg(args).query(con)?;
        Ok(info)
    }
    pub fn ai_inforeset(&self, con: &mut redis::Connection, key: String) -> RedisResult<()> {
        redis::cmd("AI.INFO").arg(key).arg("RESETSTAT").query(con)?;
        Ok(())
    }
}

#[cfg(feature = "aio")]
impl RedisAIClient {
    pub async fn ai_infoget_async(
        &self,
        con: &mut redis::aio::Connection,
        key: Option<String>,
    ) -> RedisResult<AIInfo> {
        let args = infoget_cmd_build(key);
        if self.debug {
            println!("RedisAI module running: AI.INFO {:?}", args)
        }
        let info: AIInfo = redis::cmd("AI.INFO").arg(args).query_async(con).await?;
        Ok(info)
    }
    pub async fn ai_inforeset_async(
        &self,
        con: &mut redis::aio::Connection,
        key: String,
    ) -> RedisResult<()> {
        redis::cmd("AI.INFO")
            .arg(key)
            .arg("RESETSTAT")
            .query_async(con)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ai_info_just_on_module() {
        let aiclient: RedisAIClient = RedisAIClient { debug: false };
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let mut con = client.get_connection().unwrap();
        assert_eq!(
            AIInfo {
                version: "99.99.99".to_string(),
                low_level_api_version: "1".to_string(),
                rdb_encoding_version: "1".to_string(),
            },
            aiclient.ai_infoget(&mut con, None).unwrap()
        );
    }
    //#[test]
    //#[should_panic(
    //    expected = "value: An error was signalled by the server: cannot find run info for key"
    //)]
    //fn ai_info_on_a_key() {
    //    let aiclient: RedisAIClient = RedisAIClient { debug: true };
    //    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    //    let mut con = client.get_connection().unwrap();
    //    assert_eq!(
    //        AIInfo {
    //            version: "99.99.99".to_string(),
    //            low_level_api_version: "1".to_string(),
    //            rdb_encoding_version: "1".to_string(),
    //        },
    //        aiclient
    //            .ai_infoget(&mut con, Some("m".to_string()))
    //            .unwrap()
    //    );
    //}
}
