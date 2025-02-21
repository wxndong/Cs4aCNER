from neo4j import GraphDatabase

class Neo4jConnector:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "newpassword")
        )

    def execute_query(self, query, **params):
        with self.driver.session() as session:
            return session.run(query, **params)