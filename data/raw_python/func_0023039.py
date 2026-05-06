def _process_mod(self, key, down):
        """Process (possible) keyboard modifiers

        GLFW provides "mod" with many callbacks, but not (critically) the
        scroll callback, so we keep track on our own here.
        """
        if key in MOD_KEYS:
            if down:
                if key not in self._mod:
                    self._mod.append(key)
            elif key in self._mod:
                self._mod.pop(self._mod.index(key))
        return self._mod