def get_formats(self):
        """ Return the available format names for this metadata """
        formats = []
        for key in (self.FORMAT_DC, self.FORMAT_FGDC, self.FORMAT_ISO):
            if hasattr(self, key):
                formats.append(key)
        return formats