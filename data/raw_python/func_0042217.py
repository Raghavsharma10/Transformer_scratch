def get_crop_spec(self, im, x=None, x2=None, y=None, y2=None):
        """
        Returns the default crop points for this image.
        """
        w, h = [float(v) for v in im.size]
        upscale = self.upscale
        if x is not None and x2 and y is not None and y2:
            upscale = True
            w = float(x2)-x
            h = float(y2)-y
        else:
            x = 0
            x2 = w
            y = 0
            y2 = h

        if self.width and self.height:
            ry = self.height / h
            rx = self.width / w
            if rx < ry:
                ratio = ry
                adjust = self._adjust_coordinates(ratio, w, self.width)
                x = x + adjust
                x2 = x2 - adjust
            else:
                ratio = rx
                adjust = self._adjust_coordinates(ratio, h, self.height)
                y = y + adjust
                y2 = y2 - adjust

            width = self.width
            height = self.height
        elif self.width:
            ratio = self.width / w
            width = self.width
            height = int(h * ratio)
        else:
            ratio = self.height / h
            width = int(w * ratio)
            height = self.height

        if ratio > 1 and not upscale:
            return

        x, x2, y, y2 = int(x), int(x2), int(y), int(y2)
        return CropSpec(name=self.name,
                        editable=self.editable,
                        width=width, height=height,
                        x=x, x2=x2, y=y, y2=y2)