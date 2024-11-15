import os  # To interact with the file system
import numpy as np  # Importing numpy for array operations
from tensorflow.keras.models import load_model  # To load the pre-trained model
from tensorflow.keras.preprocessing import image  # For image processing

class PredictionPipeline:
    """
    A pipeline class that loads a pre-trained model and makes predictions 
    on a provided image file.
    """
    def __init__(self, filename):
        """
        Initializes the PredictionPipeline with the image filename.
        
        Args:
            filename (str): Path to the image file to be predicted.
        """
        self.filename = filename

    def predict(self):
        """
        Loads the pre-trained model, processes the image, and returns a prediction 
        based on the model's output.
        
        Returns:
            list: A dictionary containing the image and its predicted class.
        """
        # Load the pre-trained model from the specified path
        model = load_model("artifacts/training/model.h5")
        
        # Load the image and resize it to the required input size for the model
        test_image = image.load_img(self.filename, target_size=(224, 224))
        
        # Convert the image to a numpy array and expand dimensions to match model input shape
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Get the model's prediction (probabilities) and select the class with the highest probability
        result = np.argmax(model.predict(test_image), axis=1)
        
        # Print the result for debugging purposes
        print(result)

        # Map the result to a human-readable class label
        if result[0] == 0:
            prediction = 'Cyst'
        elif result[0] == 1:
            prediction = 'Normal'
        elif result[0] == 2:
            prediction = 'Stone'            
        else:
            prediction = 'Tumor'

        # Return the prediction as a dictionary
        return [{"image": prediction}]
