def is_img(x):
    img_types = [".png", ".PNG", ".jpg", ".JPG"]
    for n in range(len(img_types)):
        if img_types[n] in x:
            return True

    return False
