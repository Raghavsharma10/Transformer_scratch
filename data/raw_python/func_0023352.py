def pop_viewport(self):
        """ Pop a viewport from the stack.
        """
        vp = self._vp_stack.pop()
        # Activate latest
        if len(self._vp_stack) > 0:
            self.context.set_viewport(*self._vp_stack[-1])
        else:
            self.context.set_viewport(0, 0, *self.physical_size)
        
        self._update_transforms()
        return vp