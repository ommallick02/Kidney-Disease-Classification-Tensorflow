import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initializes the PrepareBaseModel class with the provided configuration.

        :param config: PrepareBaseModelConfig object containing configuration parameters 
                       for base model preparation.
        """
        self.config = config

    def get_base_model(self):
        """
        Loads the VGG19 model with specified configurations, saves the model, 
        and generates a visualization of the base model.
        """
        # Load the VGG19 base model with configurations
        self.model = tf.keras.applications.vgg19.VGG19(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        # Save the base model
        self.save_model(path=self.config.base_model_path, model=self.model)

        # Create visualization directory and save base model architecture as a PNG
        visualisation_dir = "../../../visualisations"
        os.makedirs(visualisation_dir, exist_ok=True)
        plot_model(self.model, to_file=f"{visualisation_dir}/base_model.png", show_shapes=True)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate, dropout):
        """
        Prepares the full model by adding custom dense layers on top of the base model.

        :param model: The base model to be extended.
        :param classes: Number of output classes for the prediction layer.
        :param freeze_all: If True, freezes all layers of the base model.
        :param freeze_till: Number of layers from the end of the base model to keep trainable.
        :param learning_rate: Learning rate for the optimizer.
        :param dropout: Dropout rate to be applied in dense layers.

        :return: The compiled full model.
        """
        # Freeze layers of the base model as per the configurations
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Adding additional layers on top of the base model
        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        # Adding multiple dense layers with L2 regularization, batch normalization, and dropout
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(flatten_in)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        # Adding the final output layer with softmax activation for classification
        prediction = Dense(units=classes, activation="softmax")(x)

        # Create and compile the full model
        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        # Log the model summary
        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Updates the base model by adding custom layers and compiles the full model.
        Saves the updated model and generates a visualization of it.
        """
        # Prepare the full model by extending the base model
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
            dropout=self.config.params_dropout
        )

        # Save the updated model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

        # Save the architecture of the updated model as a PNG
        visualisation_dir = "../../../visualisations"
        plot_model(self.full_model, to_file=f"{visualisation_dir}/updated_base_model.png", show_shapes=True)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the given model to the specified path.

        :param path: Path where the model will be saved.
        :param model: The model to be saved.
        """
        model.save(path)
