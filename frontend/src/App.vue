<template>
  <div id="app">
    <h1>古汉语NER与知识图谱</h1>
    <router-view></router-view>
  </div>
</template>

<script>
import { createRouter, createWebHistory } from "vue-router";
import NERModule from "./components/NERModule.vue";
import KGVisualization from "./components/KGVisualization.vue";

const routes = [
  { path: "/", redirect: "/ner" }, // 默认重定向到 /ner
  { path: "/ner", component: NERModule },
  { path: "/kg", component: KGVisualization },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default {
  name: "App",
  router,
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>