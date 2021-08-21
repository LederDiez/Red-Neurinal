import os.path
import numpy as np
import tensorflow as tf
from tensorflow import keras

celsius = np.array   ([-50, -40, -10, 0, 8, 15, 22, 38, 45.5, 100, 150, 200, 250, 1000, 2001, 30004], dtype=float)
fahrenheit = np.array([-58, -40, 14, 32, 46.4, 59, 71.6, 100.4, 113.9, 212, 302, 392, 482, 1832, 3633.8, 54039.2], dtype=float)

#capa = tf.keras.layers.Dense(units=1, input_shape=[1])
#modelo = tf.keras.Sequential([capa])

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.01),
    loss='mean_squared_error'
)

if os.path.isfile('NeuronalMoldel/model.h5') is False:

    print("Comenzando entrenamiento...")
    historial = modelo.fit(celsius, fahrenheit, epochs=10000, verbose=False)
    print("Modelo entrenado!")

    modelo.save("NeuronalMoldel/model.h5")
    modelo.summary()
    
else:
    print("Cargando modelo...")
    modelo = keras.models.load_model("NeuronalMoldel/model.h5")
    modelo.summary()

def t (grados):
    resultado = modelo.predict([grados])
    valor = round(float(resultado), 1)
    print("El resultado es " + str(valor) + " fahrenheit!")
