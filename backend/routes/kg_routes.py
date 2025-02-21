from flask import Blueprint, request, jsonify
from backend.database.db_conn import get_neo4j_driver, close_neo4j_driver
import logging

logging.basicConfig(level=logging.DEBUG)  # 确保日志级别为 DEBUG

kg_bp = Blueprint("kg", __name__)

@kg_bp.route("/kg", methods=["POST"])
def search_entities():
    data = request.json
    keyword = data.get("keyword", "").strip()

    if not keyword:
        return jsonify({"error": "No keyword provided"}), 400

    driver = None
    try:
        driver = get_neo4j_driver()
        logging.info("Neo4j driver connected successfully.")
    except Exception as e:
        logging.error(f"Failed to connect to Neo4j: {e}")
        return jsonify({"error": "Database connection error"}), 500

    try:
        with driver.session() as session:
            query = """
            MATCH (n)-[r]->(m)
            WHERE n.name = $keyword
            RETURN n, r, m;
            """
            params = {"keyword": keyword} #  定义参数字典，将 keyword 变量赋值给 Cypher 查询的 $keyword 参数
            logging.debug(f"**执行的 Cypher 查询语句:**\n{query}") # 打印完整的 Cypher 查询语句
            logging.debug(f"**传递的参数:** {params}") # 打印传递的参数字典

            result = session.run(query, params) #  将 params 字典作为参数传递给 session.run()

            # 确保这行代码被 **准确地** 添加在 `result = session.run(query)` 行的 **正下方** !!!
            records = list(result)

            nodes_dict = {}
            relationships = []

            #  现在循环遍历的是 records 列表，而不是 result 对象本身
            for record in records:
                #  **新增的 debug 日志语句， 打印 record 包含的 keys**
                logging.debug(f"**Record 包含的 keys: {record.keys()}")
                #  **新增的 debug 日志语句， 打印 record['r'] 和 record['m'] 的值**
                logging.debug(f"**record['r'] 的值: {record['r']}")
                logging.debug(f"**record['m'] 的值: {record['m']}")
                # 提取源节点
                source_node = {
                    "id": record["n"].id,
                    "label": record["n"].get("name", ""),
                    "type": list(record["n"].labels)[0],
                }
                nodes_dict[source_node["id"]] = source_node

                # 提取目标节点和关系
                #  更严谨的 if 条件判断， 明确检查 record['r'] 和 record['m'] 是否为 None
                if record.get('r') is not None and record.get('m') is not None:
                    #  **新增的 debug 日志语句， 打印 if 条件为 True 的信息**
                    logging.debug(f"**[IF 条件为 True] - Record 包含 r 和 m， 开始提取关系和目标节点**")

                    target_node = {
                        "id": record["m"].id,
                        "label": record["m"].get("name", ""),
                        "type": list(record["m"].labels)[0],
                    }
                    nodes_dict[target_node["id"]] = target_node

                    relationship = {
                        "source": source_node["id"],
                        "target": target_node["id"],
                        "type": record["r"].type,
                    }
                    relationships.append(relationship)
                else:
                    #  **新增的 debug 日志语句， 打印 if 条件为 False 的信息**
                    logging.debug(f"**[IF 条件为 False] - Record **不包含** r 或 m， 跳过关系和目标节点提取**")

            # 将节点字典转换为列表
            nodes = list(nodes_dict.values())

            return jsonify({"nodes": nodes, "relationships": relationships})

    finally:
        if driver:
            close_neo4j_driver(driver)
            logging.info("Neo4j driver connection closed.")
        else:
            logging.warning("Driver was None in finally block, potential connection leak.")