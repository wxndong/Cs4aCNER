import React, { useEffect, useState } from 'react';
import axios from 'axios';

const KGDataComponent = () => {
    const [kgData, setKGData] = useState({ nodes: [], rels: [] });

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get('http://localhost:8000/get_kg_data');
                setKGData(response.data);
            } catch (error) {
                console.error('Error fetching KG data:', error);
            }
        };

        fetchData();
    }, []);

    // 这里可以添加构建图的逻辑，例如使用 vis.js 等库
    // 示例：构建节点和边的逻辑
    const buildGraph = () => {
        const nodes = new vis.DataSet(kgData.nodes.map(node => ({ id: node.id, label: node.name, group: node.type })));
        const edges = new vis.DataSet(kgData.rels.map(rel => ({ from: rel.source, to: rel.target, label: rel.rel })));

        const container = document.getElementById('kg-graph');
        const data = { nodes, edges };
        const options = {};
        new vis.Network(container, data, options);
    };

    useEffect(() => {
        if (kgData.nodes.length > 0 && kgData.rels.length > 0) {
            buildGraph();
        }
    }, [kgData]);

    return (
        <div>
            <h1>Knowledge Graph Data</h1>
            <div id="kg-graph" style={{ width: '800px', height: '600px' }}></div>
        </div>
    );
};

export default KGDataComponent;