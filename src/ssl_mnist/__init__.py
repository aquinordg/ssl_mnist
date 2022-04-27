class NormalSampling(keras.layers.Layer):
  """Implements reparametrization trick (based on VAE Keras example)."""


  def __init__(self, **kwargs):
    super(NormalSampling, self).__init__(**kwargs)


  def call(self, inputs):
    z_mean, z_log_var = inputs
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class Sunken(keras.Model):
  """Semi sUpervised NetworK ENsemble.

  Implements the ensemble of semi-supervised trained networks
  with some shared weights.
  """


  def __init__(self, shared_model_input, shared_model_output, ensemble_count, alpha=1.0, **kwargs):
    super(Sunken, self).__init__(**kwargs)

    self.x = shared_model_output
    self.z_mean = keras.layers.Dense(ensemble_count, name="z_mean")(self.x)
    self.z_log_var = keras.layers.Dense(ensemble_count, name="z_log_var")(self.x)
    self.z = NormalSampling(name="z")([self.z_mean, self.z_log_var])

    self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
    self.supervised_loss_tracker = keras.metrics.Mean(name="supervised_loss")
    self.unsupervised_loss_tracker = keras.metrics.Mean(name="unsupervised_loss")

    self.ensemble = keras.Model(
        shared_model_input, 
        [self.z, self.z_mean, self.z_log_var])
    
    self.alpha = alpha


  def summary(self, *args, **kwargs):
    self.ensemble.summary(*args, **kwargs)


  def get_layer(self, *args, **kwargs):
    return self.ensemble.get_layer(*args, **kwargs)


  def call(self, inputs):
    z, z_mean, z_log_var = self.ensemble(inputs)
    return z_mean

  
  def predict(self, inputs):
    z, z_mean, z_log_var = self.ensemble.predict(inputs)
    return z_mean  


  def score_layer(self):
    return keras.layers.Concatenate()([self.z_mean, self.z_log_var])


  @property
  def metrics(self):
    return [
        self.total_loss_tracker,
        self.supervised_loss_tracker,
        self.unsupervised_loss_tracker,
    ]


  def train_step(self, data):
    x, y = data
    with tf.GradientTape() as tape:
      z, z_mean, z_log_var = self.ensemble(x, training=True)

      supervised_loss = tf.square(y) * (tf.exp(-y*z + self.alpha) + tf.exp(z_log_var))
      unsupervised_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
      total_loss = supervised_loss + unsupervised_loss

    grads = tape.gradient(total_loss, self.ensemble.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.ensemble.trainable_weights))
    
    self.total_loss_tracker.update_state(total_loss)
    self.supervised_loss_tracker.update_state(supervised_loss)
    self.unsupervised_loss_tracker.update_state(unsupervised_loss)

    return {
      "total_loss": self.total_loss_tracker.result(),
      "supervised_loss": self.supervised_loss_tracker.result(),
      "unsupervised_loss": self.unsupervised_loss_tracker.result(),
    }
    
def prepare_sunken_multiple_output(rng, expected, ensemble_count, output_count, labeled_count):
  def binarize(y, v):
    x = np.zeros(y.shape)
    x[np.where(y != 0)[0], 0] = -1
    x[np.where(y == v)[0], 0] = 1 # order here matters, since we want to overwrite negative ones
    return x

  def sample():
    y = np.zeros((expected.shape[0], 1))
    lab = rng.choice(np.where(expected != 0)[0], labeled_count)
    y[lab, 0] = expected[lab]
    return np.concatenate([ binarize(y, i + 1) for i in range(output_count) ], axis=1)

  return np.concatenate(
      [ sample() for _ in range(ensemble_count) ], 
      axis=1)