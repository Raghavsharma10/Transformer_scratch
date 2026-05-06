def goto_next_frame(self):
        """Continue reading until the next frame is reached"""
        marked = False
        while True:
            line = next(self._f)[:-1]
            if marked and len(line) > 0 and not line.startswith(" --------"):
                try:
                    step = int(line[:10])
                    return step, line
                except ValueError:
                    pass
            marked = (len(line) == 131 and line == self._marker)