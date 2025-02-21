from neo4j import GraphDatabase

# Neo4j连接配置
URI = "bolt://localhost:7687"  # Neo4j Bolt协议地址
USER = "neo4j"  # 用户名
PASSWORD = "password"  # 密码

def get_neo4j_driver():
    return GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def close_neo4j_driver(driver):
    driver.close()