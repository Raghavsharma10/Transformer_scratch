def generate(cls, size, string, filetype="JPEG"):
        """
            Generates a squared avatar with random background color.
            :param size: size of the avatar, in pixels
            :param string: string to be used to print text and seed the random
            :param filetype: the file format of the image (i.e. JPEG, PNG)
        """
        render_size = max(size, GenAvatar.MAX_RENDER_SIZE)
        image = Image.new('RGB', (render_size, render_size),
                          cls._background_color(string))
        draw = ImageDraw.Draw(image)
        font = cls._font(render_size)
        text = cls._text(string)
        draw.text(
            cls._text_position(render_size, text, font),
            text,
            fill=cls.FONT_COLOR,
            font=font)
        stream = BytesIO()
        image = image.resize((size, size), Image.ANTIALIAS)
        image.save(stream, format=filetype, optimize=True)
        # return stream.getvalue()
        return stream