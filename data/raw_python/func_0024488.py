def aromatize(self):
        """
        convert structure to aromatic form

        :return: number of processed rings
        """
        rings = [x for x in self.sssr if 4 < len(x) < 7]
        if not rings:
            return 0
        total = 0
        while True:
            c = self._quinonize(rings, 'order')
            if c:
                total += c
            elif total:
                break

            c = self._aromatize(rings, 'order')
            if not c:
                break
            total += c

        if total:
            self.flush_cache()
        return total