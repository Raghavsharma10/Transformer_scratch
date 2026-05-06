def _generate_art(self, image, width, height):
        """
        Return an iterator that produces the ascii art.
        """
        image = image.resize((width, height), Image.ANTIALIAS).convert("RGB")
        pixels = list(image.getdata())

        for y in range(0, height - 1, 2):
            for x in range(width):
                i = y * width + x
                bg = rgb2short(*(pixels[i]))
                fg = rgb2short(*(pixels[i + width]))
                yield (fg, bg)