import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        """
        Initializes the Training class with the provided configuration.

        :param config: TrainingConfig object containing parameters for training.
        """
        self.config = config

    def get_base_model(self):
        """
        Loads the pre-trained and updated base model for further training.
        """
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):
        """
        Prepares the training and validation data generators with optional augmentation.
        """
        # Common data generator arguments
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30  # 30% data reserved for validation
        )

        # Data flow arguments
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Exclude channel dimension
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Validation data generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training data generator (with or without augmentation)
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator  # Reuse the validation generator if no augmentation

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the model to the specified path.

        :param path: Path where the model will be saved.
        :param model: The trained model instance.
        """
        model.save(path)

    def train(self):
        """
        Trains the model using the prepared data generators, applies callbacks for early stopping
        and learning rate reduction, and plots training/validation metrics.
        """
        # Define training steps
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Callbacks for training
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            verbose=0,
            mode='auto',
            min_delta=0.001,
            cooldown=0,
            min_lr=0.0
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=10,
            verbose=0,
            mode='auto',
            restore_best_weights=True
        )

        # Train the model
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            callbacks=[reduce_lr, early_stopping]
        )

        # Plot and save training metrics
        self._plot_metrics(history)

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        # Save model architecture visualization
        plot_model(self.model, to_file="../../../visualisations/trained_model.png", show_shapes=True)

    @staticmethod
    def _plot_metrics(history):
        """
        Plots and displays the training and validation accuracy/loss metrics.

        :param history: Training history object returned by `model.fit()`.
        """
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        # Plot training and validation accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.savefig("../../../visualisations/training_validation_accuracy.png")
        plt.show()

        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, 'bo-', label='Training Loss')
        plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig("../../../visualisations/training_validation_loss.png")
        plt.show()
