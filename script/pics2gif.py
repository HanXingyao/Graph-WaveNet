import os
from PIL import Image

for every_k_days in [1, 3, 5, 7]:
    directory = f'pics/every_{every_k_days}_days'

    image_files = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]

    image_files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))

    images = []

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = Image.open(image_path)
        images.append(image)

    gif_path = f'pics/every_n_days_gif/every_{every_k_days}_days.gif'
    duration = 2000
    loop = 0 

    images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=loop)
