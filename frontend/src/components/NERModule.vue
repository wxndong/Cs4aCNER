<template>
  <div id="ner-module">
    <h2>古汉语命名实体识别 (NER)</h2>
    <textarea v-model="inputText" placeholder="请输入古汉语文本..." rows="10"></textarea>
    <br />
    <button @click="submitText">提交</button>
    <div>
      <label v-for="(entity, index) in entityTypes" :key="index">
        <input type="checkbox" v-model="selectedEntities" :value="entity" /> {{ entity }}
      </label>
    </div>
    <div class="highlighted-text">
      <span v-for="(char, index) in highlightedText" :key="index" :class="char.highlight ? char.label : ''">
        {{ char.char }}
      </span>
    </div>
  </div>
</template>

<script>
import axios from "axios";

export default {
  data() {
    return {
      inputText: "",
      selectedEntities: ["NB", "NR", "NO", "NG", "NS", "T"],
      entityTypes: ["NB", "NR", "NO", "NG", "NS", "T", "O"],
      highlightedText: [],
    };
  },
  methods: {
    async submitText() {
      try {
        const response = await axios.post(
          "http://localhost:5000/api/ner",
          { text: this.inputText },
          {
            headers: {
              "Content-Type": "application/json",
            },
          }
        );
        const predictions = response.data;
        this.highlightedText = this.formatHighlightedText(predictions);
      } catch (error) {
        console.error("Error submitting text:", error);
      }
    },
    formatHighlightedText(predictions) {
      const result = [];
      predictions.forEach((item) => {
        const label = item.label;
        const baseLabel = label.split("-").pop(); // 提取后缀部分
        const shouldHighlight = this.selectedEntities.includes(baseLabel);
        result.push({ char: item.char, label: baseLabel, highlight: shouldHighlight });
      });
      return result;
    },
  },
};
</script>

<style>
.highlighted-text span.NB {
  color: blue;
}
.highlighted-text span.NR {
  color: red;
}
.highlighted-text span.NO {
  color: green;
}
.highlighted-text span.NG {
  color: orange;
}
.highlighted-text span.NS {
  color: purple;
}
.highlighted-text span.T {
  color: brown;
}
</style>