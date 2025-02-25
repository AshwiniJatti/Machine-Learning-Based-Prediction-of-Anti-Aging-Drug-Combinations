
#Importing libraries
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, LeakyReLU
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.regularizers import l1, l2
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Read data from preprocessed file from cmap(raw)
df = pd.read_csv('/disk2/user/manche/Panacea/output/cmap/merged_with_ctl_134drugs.tsv', sep='\t')
df = df.iloc[:, 2:]
print(f'df is {df.head()}')

# Normalize the data using z-score normalization
#normalized_data = (df - df.mean()) / df.std()
# Define a function for z-score normalization
def zscore(row):
    numeric_values = pd.to_numeric(row, errors='coerce')
    return (numeric_values - numeric_values.mean()) / numeric_values.std()
# Apply z-score normalization to each column
normalized_data = df.apply(zscore, axis=0)

# Reorder the columns based on the original order
normalized_data = normalized_data.loc[:, df.columns]
print(df.head())
normalized_data = normalized_data.iloc[:,0:8113]
print(f"data with cell line {normalized_data.head()}")
genes = [col for col in df.columns if any(char.isdigit() for char in col)]
#normalized_data = df
data = normalized_data[genes]
#data = np.log2(data)
print(f'data is {data.head()}')
# Extract drug names and gene expression columns
drug_names = df['cmap_name'].unique()

# Split data into train and validation sets
split_idx = int(len(data) * 0.7)
train_data = data[:split_idx]
val_data = data[split_idx:]

# Define model architecture
latent_dim = 100
input_shape = (len(genes),)
inputs = tf.keras.Input(shape=input_shape)
#print(f"inputs is {inputs}")
#x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = Dense(800)(inputs)
x = LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(rate=0.06)(x)
x = Dense(800)(x)
x = LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(rate=0.06)(x)
z_mean = tf.keras.layers.Dense(latent_dim)(x)
z_log_var = tf.keras.layers.Dense(latent_dim)(x)
#z_log_var = tf.keras.layers.LeakyReLU(alpha=0.1)(z_log_var)
z_log_var = tf.clip_by_value(z_log_var, -12, 12)

# Define sampling function for VAE
def sampling(args):
    z_mean, z_log_var = args
    #tf.print('z_log_var:', z_log_var)
    epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Define decoder layers
decoder_inputs = tf.keras.layers.Input(shape=(latent_dim,))
#x = tf.keras.layers.Dense(64, activation='relu')(decoder_inputs)
x = Dense(800)(decoder_inputs)
x = LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(rate=0.06)(x)
x = Dense(800)(x)
x = LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(rate=0.06)(x)
outputs = tf.keras.layers.Dense(len(genes))(x)

# Define VAE model
encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = tf.keras.Model(decoder_inputs, outputs, name='decoder')
vae_outputs = decoder(encoder(inputs)[2])
vae = tf.keras.Model(inputs, vae_outputs, name='vae')

# Define VAE loss function
reconstruction_loss = tf.keras.losses.mean_squared_error(inputs, vae_outputs)
#print(f"reconstruction loss is {reconstruction_loss}")
reconstruction_loss *= len(genes)
kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
#print(f"kl loss is {kl_loss}")
kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
# Compile model
#vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            experimental_run_tf_function=False)

# Define the ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, min_lr=0.00001)

# Train model
history = vae.fit(train_data, epochs=60, batch_size=80, validation_data=(val_data, None), callbacks=[reduce_lr])

# Define new input layer for combined latent vector
combined_latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))
print("defined new input layer for combined latent vector")
decoder_outputs_1 = decoder(combined_latent_inputs)
# Define new model for combined response
combined_model = tf.keras.models.Model(inputs=combined_latent_inputs, outputs=decoder_outputs_1)
print("defined combined model")

# Save the trained model
#vae.save('vae.h5')
# Save the trained model
#combined_model.save('combined_model.h5')

# Plot training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
#plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/disk2/user/manche/Panacea/output/plots/vae/loss_model.pdf', bbox_inches='tight')
plt.clf()

