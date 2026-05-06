def resize_avatar(self, img, base_width):
        """Resize an avatar.

        :param img: The image that needs to be resize.
        :param base_width: The width of output image.
        """
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        img = img.resize((base_width, h_size), PIL.Image.ANTIALIAS)
        return img