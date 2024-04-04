from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/2")

CLASS_NAMES = ['Apple___Apple_scab',
               'Apple___Black_rot',
               'Apple___Cedar_apple_rust',
               'Apple___healthy',
               'Blueberry___healthy',
               'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_',
               'Corn_(maize)___Northern_Leaf_Blight',
               'Corn_(maize)___healthy',
               'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)',
               'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)',
               'Peach___Bacterial_spot',
               'Peach___healthy',
               'Pepper,_bell___Bacterial_spot',
               'Pepper,_bell___healthy',
               'Potato___Early_blight',
               'Potato___Late_blight',
               'Potato___healthy',
               'Raspberry___healthy',
               'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch',
               'Strawberry___healthy',
               'Tomato___Bacterial_spot',
               'Tomato___Early_blight',
               'Tomato___Late_blight',
               'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']
DISEASE_SOLUTIONS = {
    "Apple___Apple_scab": "To treat Apple Scab, use fungicides like captan, myclobutanil, or mancozeb. Remove "
                          "infected leaves for better results.", "Apple___Black_rot": "For Black Rot, prune infected "
                                                                                      "parts and use fungicides such "
                                                                                      "as chlorothalonil, "
                                                                                      "pyraclostrobin, or boscalid.",
    "Apple___Cedar_apple_rust": "To control Cedar Apple Rust, use fungicides like myclobutanil, propiconazole, "
                                "or thiophanate-methyl. Remove affected branches as well.",
    "Apple___healthy": "No specific treatment needed for a healthy apple tree.",

    "Blueberry___healthy": "No specific treatment needed for healthy blueberries.",

    "Cherry_(including_sour)___healthy": "No specific treatment needed for healthy cherry trees.",
    "Cherry_(including_sour)___Powdery_mildew": "Manage Powdery Mildew in cherries with fungicides such as sulfur, "
                                                "myclobutanil, or trifloxystrobin. Prune infected parts for better "
                                                "control.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Control Cercospora Leaf Spot in corn with fungicides like "
                                                          "azoxystrobin, pyraclostrobin, or propiconazole. Implement "
                                                          "crop rotation practices.",
    "Corn_(maize)___Common_rust_": "Manage Common Rust in corn by using resistant varieties and applying fungicides "
                                   "like azoxystrobin, pyraclostrobin, or propiconazole.",
    "Corn_(maize)___healthy": "No specific treatment needed for healthy corn plants.",
    "Corn_(maize)___Northern_Leaf_Blight": "Control Northern Leaf Blight in corn with fungicides like azoxystrobin, "
                                           "pyraclostrobin, or chlorothalonil. Practice crop rotation.",

    "Grape___Black_rot": "For Black Rot in grapes, prune infected parts and use fungicides like mancozeb, captan, "
                         "or metiram.",
    "Grape___Esca_(Black_Measles)": "To manage Esca in grapes, prune infected parts and use fungicides such as "
                                    "propiconazole, trifloxystrobin, or cyprodinil.",
    "Grape___healthy": "No specific treatment needed for healthy grapevines.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Control Leaf Blight in grapes with fungicides like mancozeb, "
                                                  "copper-based fungicides, or myclobutanil. Implement proper "
                                                  "vineyard management.",

    "Orange___Haunglongbing_(Citrus_greening)": "Manage Citrus Greening by removing infected trees and controlling "
                                                "citrus psyllids. There is no cure, but antibiotics like "
                                                "oxytetracycline can be used.",

    "Peach___Bacterial_spot": "Control Bacterial Spot in peaches with copper sprays and proper sanitation. Use "
                              "copper-based fungicides like Bordeaux mixture.",
    "Peach___healthy": "No specific treatment needed for healthy peach trees.",

    "Pepper,_bell___Bacterial_spot": "Manage Bacterial Spot in bell peppers with copper sprays and disease-resistant "
                                     "varieties. Use copper-based fungicides.",
    "Pepper,_bell___healthy": "No specific treatment needed for healthy bell pepper plants.",

    "Potato___Early_blight": "Control Early Blight in potatoes with fungicides like chlorothalonil, mancozeb, "
                             "or azoxystrobin. Implement proper crop rotation.",
    "Potato___healthy": "No specific treatment needed for healthy potato plants.",
    "Potato___Late_blight": "Manage Late Blight in potatoes with fungicides like chlorothalonil, mancozeb, "
                            "or metalaxyl. Implement proper crop management practices.",

    "Raspberry___healthy": "No specific treatment needed for healthy raspberry plants.",

    "Squash___Powdery_mildew": "Control Powdery Mildew in squash with fungicides like sulfur, neem oil, or potassium "
                               "bicarbonate. Ensure proper spacing for good ventilation.",

    "Strawberry___healthy": "No specific treatment needed for healthy strawberry plants.",
    "Strawberry___Leaf_scorch": "Manage Leaf Scorch in strawberries by applying fungicides like azoxystrobin, "
                                "myclobutanil, or copper-based fungicides. Improve air circulation.",

    "Tomato___Bacterial_spot": "Control Bacterial Spot in tomatoes with copper sprays and proper plant hygiene. Use "
                               "copper-based fungicides like Bordeaux mixture.",
    "Tomato___Early_blight": "Manage Early Blight in tomatoes with fungicides like chlorothalonil, mancozeb, "
                             "or azoxystrobin. Implement proper crop rotation.",
    "Tomato___healthy": "No specific treatment needed for healthy tomato plants.",
    "Tomato___Late_blight": "Control Late Blight in tomatoes with fungicides like chlorothalonil, mancozeb, "
                            "or metalaxyl. Ensure proper moisture management.",
    "Tomato___Leaf_Mold": "Manage Leaf Mold in tomatoes with fungicides like chlorothalonil, mancozeb, "
                          "or copper-based fungicides. Ensure proper ventilation.",
    "Tomato___Septoria_leaf_spot": "Control Septoria Leaf Spot in tomatoes with fungicides like chlorothalonil, "
                                   "mancozeb, or azoxystrobin. Maintain proper plant hygiene.",
    "Tomato___Target_Spot": "Control Target Spot in tomatoes with fungicides like chlorothalonil, mancozeb, "
                            "or azoxystrobin. Ensure proper plant spacing.",
    "Tomato___Tomato_mosaic_virus": "Prevent Tomato Mosaic Virus by using disease-free seeds and controlling insect "
                                    "vectors. There is no cure for the virus.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control Tomato Yellow Leaf Curl Virus with insecticides containing "
                                              "neonicotinoids and proper vector management.",
    "Tomato___Spider_mites Two-spotted_spider_mite":
    "To control Spider Mites, use insecticidal soap, neem oil, or horticultural oil. "
    "Ensure proper plant hygiene and consider introducing natural predators like predatory mites. "
    "Regularly spray the undersides of leaves where mites often reside."
}





@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    solution = DISEASE_SOLUTIONS.get(predicted_class, "No specific solution found.")
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        
        'solution': solution
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
