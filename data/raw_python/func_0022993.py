def _new_program(self, p):
        """New program was added to the multiprogram; update items in the
        shader.
        """
        for k, v in self._set_items.items():
            getattr(p, self._shader)[k] = v