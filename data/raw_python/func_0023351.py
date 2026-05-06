def push_viewport(self, viewport):
        """ Push a viewport (x, y, w, h) on the stack. Values must be integers
        relative to the active framebuffer.

        Parameters
        ----------
        viewport : tuple
            The viewport as (x, y, w, h).
        """
        vp = list(viewport)
        # Normalize viewport before setting;
        if vp[2] < 0:
            vp[0] += vp[2]
            vp[2] *= -1
        if vp[3] < 0:
            vp[1] += vp[3]
            vp[3] *= -1

        self._vp_stack.append(vp)
        try:
            self.context.set_viewport(*vp)
        except:
            self._vp_stack.pop()
            raise
        
        self._update_transforms()