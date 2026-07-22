import os
import joblib
import gradio as gr

# Load the trained model
model = joblib.load("diabetes_prediction_model.pkl")


def predict_diabetes(pregnancies, glucose, insulin, bmi, age):
    input_data = [[pregnancies, glucose, insulin, bmi, age]]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return "🩺 Prediction: High Risk of Diabetes (Positive)"
    else:
        return "✅ Prediction: Low Risk of Diabetes (Negative)"


demo = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Age"),
    ],
    outputs=gr.Textbox(label="Assessment Result"),
    title="Diabetes Prediction System",
    description="""
Enter the medical details below to predict diabetes risk using a Decision Tree Machine Learning model.

Name: Sachin
Roll No.: 241048
""",
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
