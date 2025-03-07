import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ChemicalVAE:
    def __init__(self, input_dim, latent_dim=16, intermediate_dims=[128, 64], categorical_cols=None):
        """
        Initialize the VAE model for chemical data

        Parameters:
        -----------
        input_dim : int
            Dimensionality of the input data (number of features)
        latent_dim : int
            Dimensionality of the latent space
        intermediate_dims : list
            List of hidden layer dimensions
        categorical_cols : list
            List of categorical column names to be one-hot encoded
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_dims = intermediate_dims
        self.categorical_cols = categorical_cols or []
        
        # Preprocessing components
        self.scaler = StandardScaler()
        self.categorical_encoders = {}
        
        # Build the encoder and decoder networks
        self._build_encoder()
        self._build_decoder()
        
        # Create the VAE model
        self._build_vae()
    
    def _build_encoder(self):
        """Build the encoder network"""
        encoder_inputs = keras.Input(shape=(self.input_dim,))
        x = encoder_inputs
        
        # Add hidden layers
        for dim in self.intermediate_dims:
            x = layers.Dense(dim, activation='relu')(x)
        
        # Latent space parameters
        self.z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        self.z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Sampling layer
        self.z = Sampling()([self.z_mean, self.z_log_var])
        
        # Create encoder model
        self.encoder = keras.Model(encoder_inputs, [self.z_mean, self.z_log_var, self.z], name='encoder')
    
    def _build_decoder(self):
        """Build the decoder network"""
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = latent_inputs
        
        # Add hidden layers in reverse order
        for dim in reversed(self.intermediate_dims):
            x = layers.Dense(dim, activation='relu')(x)
        
        # Output layer
        decoder_outputs = layers.Dense(self.input_dim)(x)
        
        # Create decoder model
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
    
    def _build_vae(self):
        """Build the complete VAE model"""
        inputs = keras.Input(shape=(self.input_dim,))
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        # Define the VAE model
        self.vae = VAEModel(inputs, reconstructed, z_mean, z_log_var)
    
    def preprocess_data(self, df):
        """
        Preprocess the input DataFrame
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Preprocessed data ready for training
        """
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Handle categorical columns if any
        if self.categorical_cols:
            for col in self.categorical_cols:
                if col not in self.categorical_encoders:
                    # Initialize encoder for new categorical columns
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    # Fit the encoder
                    encoder.fit(processed_df[[col]])
                    self.categorical_encoders[col] = encoder
                
                # Transform the column
                encoded = self.categorical_encoders[col].transform(processed_df[[col]])
                # Create new column names
                encoded_cols = [f"{col}_{i}" for i in range(encoded.shape[1])]
                # Add encoded columns to dataframe
                encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=processed_df.index)
                processed_df = pd.concat([processed_df, encoded_df], axis=1)
                # Drop original categorical column
                processed_df = processed_df.drop(col, axis=1)
        
        # Convert to numpy array for scaling
        numerical_data = processed_df.values
        
        # Fit scaler if not already fit
        if not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(numerical_data)
        
        # Scale the data
        scaled_data = self.scaler.transform(numerical_data)
        
        return scaled_data
    
    def inverse_transform(self, scaled_data):
        """
        Transform scaled data back to original scale
        
        Parameters:
        -----------
        scaled_data : numpy.ndarray
            Scaled data
            
        Returns:
        --------
        numpy.ndarray
            Data in original scale
        """
        return self.scaler.inverse_transform(scaled_data)
    
    def fit(self, df, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train the VAE model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of data to use for validation
        verbose : int
            Verbosity mode (0, 1, or 2)
            
        Returns:
        --------
        keras.callbacks.History
            Training history
        """
        # Preprocess data
        processed_data = self.preprocess_data(df)
        
        # Train the model
        history = self.vae.fit(
            processed_data, 
            processed_data,  # VAE tries to reconstruct the input
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        return history
    
    def encode(self, df):
        """
        Encode data into the latent space
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Latent space representation
        """
        processed_data = self.preprocess_data(df)
        z_mean, _, _ = self.encoder.predict(processed_data)
        return z_mean
    
    def decode(self, latent_vectors):
        """
        Decode from latent space to original feature space
        
        Parameters:
        -----------
        latent_vectors : numpy.ndarray
            Latent space vectors
            
        Returns:
        --------
        numpy.ndarray
            Reconstructed data in original scale
        """
        reconstructed_scaled = self.decoder.predict(latent_vectors)
        return self.inverse_transform(reconstructed_scaled)
    
    def save(self, filepath):
        """Save the model to a file"""
        self.vae.save(filepath)
    
    def load(self, filepath):
        """Load the model from a file"""
        self.vae = keras.models.load_model(
            filepath,
            custom_objects={
                'VAEModel': VAEModel,
                'Sampling': Sampling
            }
        )
        # Extract encoder and decoder from the loaded model
        self.encoder = self.vae.get_layer('encoder')
        self.decoder = self.vae.get_layer('decoder')


class Sampling(layers.Layer):
    """
    Custom sampling layer for VAE
    Uses reparameterization trick: z = mean + exp(0.5 * log_var) * epsilon
    where epsilon is a random normal tensor
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEModel(keras.Model):
    """
    Custom Keras Model for VAE with appropriate loss function
    """
    def __init__(self, inputs, reconstructed, z_mean, z_log_var, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.inputs = inputs
        self.reconstructed = reconstructed
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        
        # Create the model
        self.vae_net = keras.Model(inputs, reconstructed)
        
        # Add loss
        self.add_loss(self._get_loss())
        
        # Configure optimizer
        self.compile(optimizer=optimizers.Adam())
    
    def _get_loss(self):
        """Compute VAE loss: reconstruction loss + KL divergence"""
        # Reconstruction loss (mean squared error)
        reconstruction_loss = tf.reduce_mean(
            losses.mean_squared_error(self.inputs, self.reconstructed)
        )
        
        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var)
        )
        
        # Total loss
        return reconstruction_loss + kl_loss
    
    def call(self, inputs):
        """Forward pass through the model"""
        return self.vae_net(inputs)
    
    def train_step(self, data):
        """Custom training step"""
        if isinstance(data, tuple):
            data = data[0]
        
        with tf.GradientTape() as tape:
            reconstructed = self(data, training=True)
            loss = self.compiled_loss(data, reconstructed, regularization_losses=self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {'loss': loss}


# Example usage
if __name__ == "__main__":
    # Load your data
    # df = pd.read_csv('your_data.csv')
    
    # For demonstration, we'll create a sample DataFrame similar to your structure
    np.random.seed(42)
    sample_data = {
        'api': np.random.randint(2200, 2300, 1000),
        'solvent': np.random.randint(700, 710, 1000),
        'solubility': np.random.random(1000),
        'fraction': np.random.random(1000),
        'temperature': np.random.random(1000) * 0.3,
        'AM1_dipole_x': np.random.random(1000),
        'vsurf_Wp5_y': np.random.random(1000) * 0.2,
        'vsurf_Wp6_y': np.random.random(1000) * 0.2,
        'Weight_y': np.random.random(1000) * 0.1,
        'weinerPath_y': np.zeros(1000),
        'weinerPol_y': np.zeros(1000),
        'zagreb_y': np.zeros(1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Define categorical columns (if any)
    categorical_cols = ['api', 'solvent']
    
    # Calculate input dimension (for numeric data after one-hot encoding)
    # For demonstration, we'll assume all columns except categorical ones are numeric
    numeric_cols = [col for col in df.columns if col not in categorical_cols]
    
    # Initialize the VAE model
    vae = ChemicalVAE(
        input_dim=len(numeric_cols),  # This will be updated after preprocessing
        latent_dim=8,
        intermediate_dims=[64, 32],
        categorical_cols=categorical_cols
    )
    
    # Preprocess a batch to determine the actual input dimension after one-hot encoding
    processed_batch = vae.preprocess_data(df.sample(10))
    actual_input_dim = processed_batch.shape[1]
    
    # Reinitialize with correct input dimension
    vae = ChemicalVAE(
        input_dim=actual_input_dim,
        latent_dim=8,
        intermediate_dims=[64, 32],
        categorical_cols=categorical_cols
    )
    
    # Train the model
    history = vae.fit(df, epochs=50, batch_size=32, verbose=1)
    
    # Encode data to latent space
    latent_representations = vae.encode(df)
    
    # Visualize the latent space (first 2 dimensions)
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], 
                c=df['solubility'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='Solubility')
    plt.title('VAE Latent Space Visualization')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.tight_layout()
    plt.savefig('vae_latent_space.png')
    
    # Generate new samples
    random_latent_points = np.random.normal(size=(10, vae.latent_dim))
    generated_samples = vae.decode(random_latent_points)
    
    print("Generated samples in original feature space:")
    print(generated_samples[:5])
    
    # Save the model
    vae.save('chemical_vae_model')
    
    print("VAE model training and evaluation completed!")


# Function to create a more comprehensive VAE training and evaluation pipeline
def train_chemical_vae(df, latent_dim=16, epochs=100, batch_size=32):
    """
    Train a VAE model on chemical data and evaluate its performance
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    latent_dim : int
        Dimension of the latent space
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    ChemicalVAE
        Trained VAE model
    """
    # Identify likely categorical columns (those with few unique values)
    categorical_threshold = 20  # Columns with fewer unique values are treated as categorical
    likely_categorical = [col for col in df.columns 
                         if df[col].nunique() < categorical_threshold and df[col].dtype != float]
    
    print(f"Identified categorical columns: {likely_categorical}")
    
    # Split data for training and testing
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Calculate input dimension
    numeric_cols = [col for col in df.columns if col not in likely_categorical]
    
    # Initialize VAE with placeholder input dimension
    vae = ChemicalVAE(
        input_dim=len(numeric_cols),
        latent_dim=latent_dim,
        intermediate_dims=[128, 64],
        categorical_cols=likely_categorical
    )
    
    # Preprocess a batch to determine the actual input dimension after one-hot encoding
    processed_batch = vae.preprocess_data(train_df.sample(min(10, len(train_df))))
    actual_input_dim = processed_batch.shape[1]
    
    print(f"Actual input dimension after preprocessing: {actual_input_dim}")
    
    # Reinitialize with correct input dimension
    vae = ChemicalVAE(
        input_dim=actual_input_dim,
        latent_dim=latent_dim,
        intermediate_dims=[128, 64],
        categorical_cols=likely_categorical
    )
    
    # Print model summary
    print("Encoder model summary:")
    vae.encoder.summary()
    
    print("\nDecoder model summary:")
    vae.decoder.summary()
    
    # Train the model
    history = vae.fit(train_df, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('VAE Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('vae_training_history.png')
    
    # Evaluate reconstruction performance
    test_processed = vae.preprocess_data(test_df)
    test_encoded = vae.encoder.predict(test_processed)[0]  # Get z_mean
    test_reconstructed = vae.decoder.predict(test_encoded)
    
    # Calculate reconstruction error
    mse = np.mean(np.square(test_processed - test_reconstructed))
    print(f"Mean Squared Error on test set: {mse:.6f}")
    
    # Visualize latent space
    train_encoded = vae.encode(train_df)
    
    # Create a scatter plot of the first two latent dimensions
    plt.figure(figsize=(12, 10))
    
    # Try to color by a meaningful feature if available
    if 'solubility' in train_df.columns:
        color_feature = 'solubility'
    elif 'temperature' in train_df.columns:
        color_feature = 'temperature'
    else:
        color_feature = train_df.columns[0]
    
    scatter = plt.scatter(
        train_encoded[:, 0], 
        train_encoded[:, 1],
        c=train_df[color_feature], 
        cmap='viridis', 
        alpha=0.7,
        s=50
    )
    
    plt.colorbar(scatter, label=color_feature)
    plt.title(f'VAE Latent Space (colored by {color_feature})')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('vae_latent_space_colored.png')
    
    # Generate new samples
    print("\nGenerating new samples from the latent space:")
    random_latent_points = np.random.normal(size=(5, latent_dim))
    generated_samples = vae.decode(random_latent_points)
    
    # Return the trained model
    return vae