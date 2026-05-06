def from_string(cls, width, height, rgba_string):
        """Returns a Form with 32-bit RGBA pixels
        Accepts string containing raw RGBA color values
        """
        # Convert RGBA string to ARGB
        raw = ""
        for i in range(0, len(rgba_string), 4):
            raw += rgba_string[i+3]   # alpha
            raw += rgba_string[i:i+3] # rgb

        assert len(rgba_string) == width * height * 4

        return Form(
            width = width,
            height = height,
            depth = 32,
            bits = Bitmap(raw),
        )