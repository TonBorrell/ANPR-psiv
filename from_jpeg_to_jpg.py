from PIL import Image

for i in range(9, 17):
    image = "images/good/4773_LGG.HEIC".format(i)
    img = Image.open(image)
    rgb_img = img.convert("RGB")
    rgb_img.save("images/good/4773_LGG.jpg".format(i))
