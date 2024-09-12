import numpy as np
import librosa.display, os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import MobileNetV2


# Load the saved model
model = load_model('model.h5')
print(model.input_shape)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)
    
def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)


# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to predict sound type from audio file
def predict_sound_type(audio_file):
    # Step 1: Create a spectrogram
    spectrogram_path = 'temp_spectrogram.png'
    create_spectrogram(audio_file, spectrogram_path)
    
    # Step 2: Preprocess the image (spectrogram)
    img = preprocess_image(spectrogram_path)
    
    # Step 3: Extract features using MobileNetV2
    features = base_model.predict(img)
    
    # Step 4: Make a prediction using your trained model
    predictions = model.predict(features)
    
    # Labels for prediction (based on your training labels)
    class_labels = ['dog_bark', 'children_playing', 'drilling', 'gun_shot']
    
    # Step 5: Print predictions
    for i, label in enumerate(class_labels):
        print(f'{label}: {predictions[0][i]*100:.2f}%')

# Example: Using the model to predict the sound in a WAV file
# predict_sound_type('20285-3-1-1.wav')


