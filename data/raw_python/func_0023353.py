def push_fbo(self, fbo, offset, csize):
        """ Push an FBO on the stack.
        
        This activates the framebuffer and causes subsequent rendering to be
        written to the framebuffer rather than the canvas's back buffer. This
        will also set the canvas viewport to cover the boundaries of the 
        framebuffer.

        Parameters
        ----------
        fbo : instance of FrameBuffer
            The framebuffer object .
        offset : tuple
            The location of the fbo origin relative to the canvas's framebuffer
            origin.
        csize : tuple
            The size of the region in the canvas's framebuffer that should be 
            covered by this framebuffer object.
        """
        self._fb_stack.append((fbo, offset, csize))
        try:
            fbo.activate()
            h, w = fbo.color_buffer.shape[:2]
            self.push_viewport((0, 0, w, h))
        except Exception:
            self._fb_stack.pop()
            raise
        
        self._update_transforms()