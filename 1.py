import numpy as np
import librosa.display, os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pydub import AudioSegment


# Load the saved model
model = load_model('model.h5')
# print(model.input_shape)
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

    # Return prediction for dog barking
    return predictions[0][0]  # Dog barking probability

# Example: Using the model to predict the sound in a WAV file
# predict_sound_type('20285-3-1-1.wav')



# Function to check and record video segments if dog barking is detected
def monitor_video(input_video):
    audio = AudioSegment.from_file(input_video, format="mp4")
    chunk_size_ms = 4000  # 4 seconds in milliseconds
    last_saved_time = None
    
    for i in range(0, len(audio), chunk_size_ms):
        # Extract 4-second audio chunk
        chunk = audio[i:i + chunk_size_ms]
        chunk.export("temp_audio.wav", format="wav")
        
        # Predict the sound type
        is_dog_bark = predict_sound_type("temp_audio.wav") > 0.9  # Threshold 90%

        # Calculate the start time (in seconds)
        start_time = i / 1000
        
        # Check if we need to save the video
        if is_dog_bark:
            if last_saved_time is None or start_time - last_saved_time > 20:
                # Save the video clip if no video was saved in the last minute
                print(f"Dog barking detected at {start_time:.2f}s, saving video...")
                save_video_clip(input_video, start_time, 4)
                last_saved_time = start_time
            else:
                print(f"Dog barking detected at {start_time:.2f}s, but within 20 seconds of last clip. Skipping.")

# Function to save a 4-second video clip
def save_video_clip(input_video, start_time, duration):
    output_file = f"dog_bark_{start_time:.2f}.mp4"
    ffmpeg_extract_subclip(input_video, start_time, start_time + duration, targetname=output_file)
    print(f"Saved video clip: {output_file}")

# Start monitoring the video
monitor_video('dog.mp4')
# monitor_video('1.mp4')