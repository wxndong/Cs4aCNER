from fastapi import FastAPI
from neo4j import GraphDatabase

app = FastAPI()
driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "password"))

@app.get("/get_kg_data")
async def get_kg_data():
    # 查询节点数据
    nodes_cypher = "MATCH (n) RETURN n.id as id, n.name as name, n.type as type"
    with driver.session() as session:
        nodes_result = session.run(nodes_cypher)
        nodes = [dict(record) for record in nodes_result]

    # 查询关系数据
    rels_cypher = "MATCH (n)-[r]->(m) RETURN n.id as source, m.id as target, type(r) as rel"
    with driver.session() as session:
        rels_result = session.run(rels_cypher)
        rels = [dict(record) for record in rels_result]

    return {"nodes": nodes, "rels": rels}