def _generate_art(self, image, width, height):
        """
        Return an iterator that produces the ascii art.
        """
        # Characters aren't square, so scale the output by the aspect ratio of a charater
        height = int(height * self._char_width / float(self._char_height))
        image = image.resize((width, height), Image.ANTIALIAS).convert("RGB")

        for (r, g, b) in image.getdata():
            greyscale = int(0.299 * r + 0.587 * g + 0.114 * b)
            ch = self._chars[int(greyscale / 255. * (len(self._chars) - 1) + 0.5)]
            yield (ch, rgb2short(r, g, b))