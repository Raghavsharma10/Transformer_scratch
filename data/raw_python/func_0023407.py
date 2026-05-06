def get_font(self, face, bold=False, italic=False):
        """Get a font described by face and size"""
        key = '%s-%s-%s' % (face, bold, italic)
        if key not in self._fonts:
            font = dict(face=face, bold=bold, italic=italic)
            self._fonts[key] = TextureFont(font, self._renderer)
        return self._fonts[key]