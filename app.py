import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Set environment variables for language settings
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for all routes

# ClientApp class to manage file and classifier initialization
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"  # Default filename for input images
        self.classifier = PredictionPipeline(self.filename)  # Initialize prediction pipeline with filename

# Route for the homepage
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')  # Render the index.html page on homepage

# Route for model training
@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    # Run DVC pipeline to retrain the model
    os.system("dvc repro")  # Executes the command to retrain via DVC
    return "Training done successfully!"

# Route for making predictions
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    # Decode the incoming image data and save to the specified file
    image = request.json['image']
    decodeImage(image, clApp.filename)
    
    # Make prediction using the classifier
    result = clApp.classifier.predict()
    return jsonify(result)  # Return the prediction result as JSON

if __name__ == "__main__":
    clApp = ClientApp()  # Instantiate the ClientApp class
    app.run(host='0.0.0.0', port=8080)  # Run the app, making it accessible from any network interface
