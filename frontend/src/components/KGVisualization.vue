<template>
  <div>
    <h2>古汉语NER与知识图谱</h2>
    <input v-model="keyword" placeholder="输入关键词搜索..." />
    <button @click="searchEntities">搜索</button>
    <svg ref="graph" width="800" height="600"></svg>
  </div>
</template>

<script>
import * as d3 from "d3";
import axios from "axios";

export default {
  data() {
    return {
      keyword: "",
      graphData: { nodes: [], relationships: [] },
    };
  },
  mounted() {
    this.initZoom();
  },
  methods: {
    async searchEntities() {
      try {
        const response = await axios.post("http://localhost:5000/api/kg", { keyword: this.keyword });
        this.graphData = response.data;
        this.renderGraph();
      } catch (error) {
        console.error("Error fetching KG data:", error);
      }
    },
    renderGraph() {
      const svg = d3.select(this.$refs.graph);
      svg.selectAll("*").remove();

      const width = 800;
      const height = 600;

      // Force layout parameters - Adjusted for better spacing
      const simulation = d3
          .forceSimulation(this.graphData.nodes)
          .force("link", d3.forceLink(this.graphData.relationships).id((d) => d.id).distance(150)) // Increased link distance
          .force("charge", d3.forceManyBody().strength(-300)) // Increased charge strength for more repulsion
          .force("center", d3.forceCenter(width / 2, height / 2))
          .force("collide", d3.forceCollide().radius(d => 20).iterations(2)); // Added collision force

      const link = svg
          .append("g")
          .attr("class", "links")
          .selectAll("line")
          .data(this.graphData.relationships)
          .enter()
          .append("line")
          .attr("stroke", "#c0c0c0") // Lighter gray for links
          .attr("stroke-width", 1.5);

      const linkLabels = svg.select(".links")
          .selectAll("text")
          .data(this.graphData.relationships)
          .enter()
          .append("text")
          .attr("class", "link-label")
          .text(d => d.type)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle");

      const node = svg
          .append("g")
          .attr("class", "nodes")
          .selectAll("circle")
          .data(this.graphData.nodes)
          .enter()
          .append("circle")
          .attr("r", 18) // Increased node radius even more
          .attr("fill", " SteelBlue ") // More distinct node color
          .attr("stroke", "#fff") // White node stroke
          .attr("stroke-width", 2)
          .call(drag(simulation));

      const text = svg
          .append("g")
          .selectAll("text")
          .data(this.graphData.nodes)
          .enter()
          .append("text")
          .attr("class", "node-label")
          .text((d) => d.label)
          .attr("x", 22) // Increased horizontal offset for labels
          .attr("y", 4)  // Slight vertical adjustment for labels
          .style("font-size", "15px") // Slightly larger font size for labels
          .attr("fill", "#333"); // Darker label color


      simulation.on("tick", () => {
        link
            .attr("x1", (d) => d.source.x)
            .attr("y1", (d) => d.source.y)
            .attr("x2", (d) => d.target.x)
            .attr("y2", (d) => d.target.y);

        linkLabels
            .attr("x", d => (d.source.x + d.target.x) / 2)
            .attr("y", d => (d.source.y + d.target.y) / 2);

        node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

        text.attr("x", (d) => d.x).attr("y", (d) => d.y);
      });

      function drag(simulation) {
        function dragstarted(event, d) {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        }

        function dragged(event, d) {
          d.fx = event.x;
          d.fy = event.y;
        }

        function dragended(event, d) {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }

        return d3
            .drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
      }
    },
    initZoom() {
      const svg = d3.select(this.$refs.graph);
      const zoomBehavior = d3.zoom()
          .scaleExtent([0.1, 4])
          .on("zoom", (event) => {
            svg.attr("transform", event.transform);
          });

      svg.call(zoomBehavior);
    },
  },
};
</script>

<style scoped>
svg {
  border: 1px solid #ccc;
  user-select: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
}

svg * {
  user-select: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
}

.links line {
  stroke-opacity: 0.8; /* Slightly increased link opacity */
}

.link-label {
  font-size: 12px; /* Slightly larger link label font */
  fill: #555;
  text-shadow: 1px 1px 0 #fff;
  /* background-color: rgba(255, 255, 255, 0.7);  Optional background for link labels */
  padding: 2px 4px; /* Add padding to link labels */
  border-radius: 3px; /* Add border radius to link labels */
}


.nodes circle {
  stroke-width: 2px;
}

.node-label {
  font-weight: bold; /* Bold font for node labels */
  text-shadow: 1px 1px 0 #fff; /* White text shadow for better contrast */
}

</style>