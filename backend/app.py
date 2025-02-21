from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 注册路由
from routes.ner_routes import ner_bp
from routes.kg_routes import kg_bp
app.register_blueprint(ner_bp, url_prefix="/api")
app.register_blueprint(kg_bp, url_prefix="/api")

@app.before_request
def handle_options():
    """
    处理预检请求（OPTIONS 方法）
    """
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

@app.route('/')
def home():
    return "CS4ACNER Backend Service is Running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)