import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
import os
from datetime import datetime

# Create directory for saving plots
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
plot_dir = f'gan_plots_{timestamp}'
os.makedirs(plot_dir, exist_ok=True)

# Data generation functions remain the same
def get_y(x):
    return 10 + x*x

def sample_data(n=10000, scale=100):
    data = []
    x = scale*(np.random.random_sample((n,))-0.5)
    
    for i in range(n):
        yi = get_y(x[i])
        data.append([x[i], yi])
        
    return np.array(data)

def sample_Z(batch_size, dim):
    return np.random.uniform(-1., 1., size=[batch_size, dim])

# Modified visualization function to save plots
def plot_results(generator, iteration, n_samples=1000, save=True, show=True):
    # Generate samples
    z_noise = sample_Z(n_samples, 2)
    generated_samples = generator(z_noise).numpy()
    
    # Generate real samples for comparison
    real_samples = sample_data(n=n_samples)
    
    plt.figure(figsize=(12, 8))
    
    # Plot real data
    plt.scatter(real_samples[:, 0], real_samples[:, 1], c='blue', alpha=0.2, label='Real Data')
    
    # Plot generated data
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], c='red', alpha=0.2, label='Generated Data')
    
    plt.title(f'Real vs Generated Data (Iteration {iteration})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    if save:
        filename = os.path.join(plot_dir, f'gan_plot_iteration_{iteration}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    if show:
        plt.show()
    
    plt.close()

# Generator and Discriminator classes remain the same
class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = layers.Dense(128, activation='leaky_relu')
        self.layer2 = layers.Dense(128, activation='leaky_relu')
        self.output_layer = layers.Dense(2)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return self.output_layer(x)

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = layers.Dense(128, activation='leaky_relu')
        self.layer2 = layers.Dense(128, activation='leaky_relu')
        self.feature_layer = layers.Dense(2)
        self.output_layer = layers.Dense(1)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        features = self.feature_layer(x)
        output = self.output_layer(features)
        return output, features

# Training function remains the same
@tf.function
def train_step(real_data, noise):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)
        real_output, _ = discriminator(real_data, training=True)
        fake_output, _ = discriminator(generated_data, training=True)
        
        gen_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                tf.ones_like(fake_output), fake_output, from_logits=True
            )
        )
        
        disc_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                tf.ones_like(real_output), real_output, from_logits=True
            ) +
            tf.keras.losses.binary_crossentropy(
                tf.zeros_like(fake_output), fake_output, from_logits=True
            )
        )
    
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Create models and optimizers
generator = Generator()
discriminator = Discriminator()
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training parameters
batch_size = 256
n_iterations = 50001

# Save loss history
loss_history = {
    'generator_loss': [],
    'discriminator_loss': [],
    'iterations': []
}

# Training loop
for i in range(n_iterations):
    X_batch = sample_data(n=batch_size)
    Z_batch = sample_Z(batch_size, 2)
    
    g_loss, d_loss = train_step(X_batch, Z_batch)
    
    # Store losses
    if i % 100 == 0:  # Store more frequently for better plots
        loss_history['generator_loss'].append(float(g_loss))
        loss_history['discriminator_loss'].append(float(d_loss))
        loss_history['iterations'].append(i)
    
    if i % 1000 == 0:
        print(f"Iteration: {i}\tDiscriminator loss: {d_loss:.4f}\tGenerator loss: {g_loss:.4f}")
        
        if i % 10000 == 0:
            # Save plot but don't show it during training
            plot_results(generator, i, show=False)

# Plot and save final results
plot_results(generator, n_iterations-1)

# Plot and save loss history
plt.figure(figsize=(12, 6))
plt.plot(loss_history['iterations'], loss_history['generator_loss'], label='Generator Loss')
plt.plot(loss_history['iterations'], loss_history['discriminator_loss'], label='Discriminator Loss')
plt.title('Training Loss Over Time')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Log scale for better visualization
filename = os.path.join(plot_dir, 'loss_history.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

# Save final model
model_dir = os.path.join(plot_dir, 'final_model')
os.makedirs(model_dir, exist_ok=True)
generator.save(os.path.join(model_dir, 'generator.keras'))
discriminator.save(os.path.join(model_dir, 'discriminator.keras'))


generator.save_weights(os.path.join(model_dir, 'generator_weights.keras'))
discriminator.save_weights(os.path.join(model_dir, 'discriminator_weights.keras'))


print(f"\nTraining completed. All results saved in directory: {plot_dir}")