import cv2
import numpy as np
import requests
import telegram
from io import BytesIO

# Set up the Telegram bot
bot = telegram.Bot("5188076057:AAEz1nkWXHmdipjZz_5S_NzMd5lPwD5SqN0")

def convert_to_anime(image_url):
    # Download the image from the URL
    response = requests.get(image_url)
    image = np.array(bytearray(response.content), dtype=np.uint8)

    # Load the Deep Learning model
    model = cv2.dnn.readNetFromCaffe("anime_style_transfer.prototxt", "anime_style_transfer.caffemodel")

    # Pre-process the image for the model
    blob = cv2.dnn.blobFromImage(image, 1.0, (256, 256), (103.939, 116.779, 123.68), swapRB=False, crop=False)

    # Feed the blob through the model to get the output
    model.setInput(blob)
    output = model.forward()

    # Post-process the output to get the final image
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.68
    output /= 255.0
    output = output.transpose(1, 2, 0)

    # Save the output image to a BytesIO object
    output_buffer = BytesIO()
    cv2.imwrite(output_buffer, output)

    return output_buffer

def handle_message(message):
    # Check if the message contains an image
    if message.photo:
        # Get the largest size of the image
        image_file = bot.get_file(message.photo[-1].file_id)
        image_url = image_file.file_path

        # Convert the image to anime
        output_buffer = convert_to_anime(image_url)

        # Send the output image back to the user
        bot.send_photo(message.chat.id, photo=output_buffer)

# Start listening for incoming messages
updates = bot.get_updates()
while True:
    try:
        # Get the next update
        update = bot.get_update()
        message = update.message
        handle_message(message)
    except Exception as e:
        print(e)
