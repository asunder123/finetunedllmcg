import vertexai
from vertexai.preview.language_models import TextGenerationModel

vertexai.init(project="280303677176", location="us-central1")
parameters = {
    "temperature": 0.2,
    "max_output_tokens": 256,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison@001")
model = model.get_tuned_model("projects/280303677176/locations/us-central1/models/7578953441469267968")
response = model.predict(
    """As an investor hoping for a 5% YoY growth for contingency savings which should i invest  in based on tuned model """,
    **parameters
)
print(f"Response from Model: {response.text}")
