from flask import Flask, render_template, request, jsonify
from vertexai.preview.language_models import TextGenerationModel

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        parameters = {
            "temperature": 0.2,
            "max_output_tokens": 256,
            "top_p": 0.8,
            "top_k": 40
        }
        model = TextGenerationModel.from_pretrained("text-bison@001")
        model = model.get_tuned_model("projects/280303677176/locations/us-central1/models/7578953441469267968")
        response = model.predict(prompt, **parameters)
        return jsonify({"response": response.text})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

