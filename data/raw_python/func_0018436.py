def shawn_text(self):
        """The text displayed instead of the real one."""

        if len(self._shawn_text) == len(self):
            return self._shawn_text

        if self.style == self.DOTS:
            return chr(0x2022) * len(self)

        ranges = [
            (902, 1366),
            (192, 683),
            (33, 122)
        ]

        s = ''
        while len(s) < len(self.text):
            apolo = randint(33, 1366)
            for a, b in ranges:
                if a <= apolo <= b:
                    s += chr(apolo)
                    break

        self._shawn_text = s
        return s