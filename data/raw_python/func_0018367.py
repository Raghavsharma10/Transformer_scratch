def get_darker_image(self):
        """Returns an icon 80% more dark"""
        icon_pressed = self.icon.copy()

        for x in range(self.w):
            for y in range(self.h):
                r, g, b, *_ = tuple(self.icon.get_at((x, y)))
                const = 0.8
                r = int(const * r)
                g = int(const * g)
                b = int(const * b)
                icon_pressed.set_at((x, y), (r, g, b))

        return icon_pressed