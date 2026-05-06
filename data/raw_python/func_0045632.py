def state(self):
        """Returns a new JIT state. You have to clean up by calling .destroy()
        afterwards.
        """
        return Emitter(weakref.proxy(self.lib), self.lib.jit_new_state())