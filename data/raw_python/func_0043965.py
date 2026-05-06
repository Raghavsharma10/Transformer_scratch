def check_animated(cls, raw_image):
        "Checks whether the gif is animated."
        try:
            raw_image.seek(1)
        except EOFError:
            isanimated= False
        else:
            isanimated= True
            raise cls.AnimatedImageException