def layered(self):
        """Yield list of [[(name, image), ...], [(name, image), ...], ...]"""
        result = []
        for layer in self._layered:
            nxt = []
            for name in layer:
                nxt.append((name, self.all_images[name]))
            result.append(nxt)
        return result