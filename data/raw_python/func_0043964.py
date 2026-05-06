def check_color(cls, raw_image):
        """
        Just check if raw_image is completely white.
        http://stackoverflow.com/questions/14041562/python-pil-detect-if-an-image-is-completely-black-or-white
        """
        # sum(img.convert("L").getextrema()) in (0, 2)
        extrema = raw_image.convert("L").getextrema()
        if extrema == (255, 255): # all white
            raise cls.MonoImageException