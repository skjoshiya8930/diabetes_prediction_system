import os
import joblib
import gradio as gr

# Load the trained model
model = joblib.load("diabetes_prediction_model.pkl")


def predict_diabetes(pregnancies, glucose, insulin, bmi, age):
    try:
        # Convert inputs to float
        input_data = [[
            float(pregnancies),
            float(glucose),
            float(insulin),
            float(bmi),
            float(age)
        ]]

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            return "🩺 Prediction: High Risk of Diabetes (Positive)"
        else:
            return "✅ Prediction: Low Risk of Diabetes (Negative)"

    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(title="Diabetes Prediction System") as demo:
    gr.Markdown(
        """
        # 🩺 Diabetes Prediction System

        Enter the medical details below to predict diabetes risk using a Decision Tree Machine Learning model.

        **Name:** Sachin  
        **Roll No.:** 241048
        """
    )

    pregnancies = gr.Number(label="Pregnancies")
    glucose = gr.Number(label="Glucose")
    insulin = gr.Number(label="Insulin")
    bmi = gr.Number(label="BMI")
    age = gr.Number(label="Age")

    output = gr.Textbox(label="Prediction")

    predict_btn = gr.Button("Predict")

    predict_btn.click(
        fn=predict_diabetes,
        inputs=[pregnancies, glucose, insulin, bmi, age],
        outputs=output,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
