def _char_density(self, c, font=ImageFont.load_default()):
        """
        Count the number of black pixels in a rendered character.
        """
        image = Image.new('1', font.getsize(c), color=255)
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), c, fill="white", font=font)
        return collections.Counter(image.getdata())[0]