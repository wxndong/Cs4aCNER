import { createRouter, createWebHistory } from "vue-router";
import NERModule from "../components/NERModule.vue";
import KGVisualization from "../components/KGVisualization.vue";

const routes = [
  { path: "/", redirect: "/ner" }, // 默认重定向到 /ner
  { path: "/ner", component: NERModule },
  { path: "/kg", component: KGVisualization },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;