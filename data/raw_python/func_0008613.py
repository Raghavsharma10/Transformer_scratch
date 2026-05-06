def stringify(self):
        """Returns the color value in hexcode format.

        eg. ``'#ff1056'``

        """
        hexcode = "#"
        for x in self.value:
            part = hex(x)[2:]
            if len(part) < 2: part = "0" + part
            hexcode += part
        return hexcode