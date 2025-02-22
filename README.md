#
# 项目计划
```mermaid
gantt
    title 消融实验执行计划
    dateFormat  YYYY-MM-DD
    section 原型搭建
    NER模块 + KG模块       :active, des1, 2025-02-20, 2d
    LLM-QA模块              :crit, des2, after des1, 2d
    数据库搭建              :des7, after des2, 2d
    section 功能完善
    NER 模块优化             :active, des3, after des7, 2d
    LLM-QA模块优化          :crit, des4, after des3, 2d
    section 收尾阶段
    Docker部署到服务器       :des5, after des4, 2d
    文档技术总结             :des6, after des5, 2d
```

# 交互设计
```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant NER_API
    participant KG_DB
    participant LLM_API

    User->>Frontend: 输入文本
    Frontend->>NER_API: 发送文本
    NER_API-->>Frontend: 返回实体列表
    Frontend->>User: 展示实体列表

    User->>Frontend: 选择实体并提问
    Frontend->>KG_DB: 查询实体关联数据
    KG_DB-->>Frontend: 返回子图信息
    Frontend->>LLM_API: 组合提问+KG数据
    LLM_API-->>Frontend: 生成回答
    Frontend->>User: 显示最终回答
```
# TO DO      
- 下一步：联合测试neo4j的前后端页面及数据库，实现：
  - [x] 编写基础的前后端逻辑
  - [ ] docker启动neo4j、后端、前端
  - [x] 后端API接口连接上数据库
  - [x] 前端调用后端的API数据进行渲染并查看
  - [x] NER的标签错了，记得改
  - [ ] 大模型API引入
  - [ ] 三个模块之间建立关联
  - [ ] 界面功能美化



---
# Time Line
## 2025.2.21 下午
模块1、2的前后端雏形完成了，现在需要安排上docker，否则每次启动太墨迹。

## 2025.2.21 上午
- 搭建了基本的项目框架： 前端、后端、数据库
  - 单独测试了使用docker启动neo4j，并查看;
    - neo4j新版本登录选择 `bold + localhost:7687`
    - windows挂载文件时似乎有问题，应该用绝对路径，并且neo4j命令也是如此
    - ```bash
        LOAD CSV WITH HEADERS FROM 'file:///var/lib/neo4j/import/nodes.csv' AS row
        MERGE (:Node {id: row.id, name: row.name, type: row.type});
      ```
    - ```bash
        LOAD CSV WITH HEADERS FROM 'file:///var/lib/neo4j/import/relationships.csv' AS rel_row
        WITH rel_row
        MATCH (source:Node {id: rel_row.source})
        WITH source, rel_row
        MATCH (target:Node {id: rel_row.target})
        CALL apoc.create.relationship(source, rel_row.rel, {}, target) YIELD rel
        RETURN rel;
      ```  
