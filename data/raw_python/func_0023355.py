def _update_transforms(self):
        """Update the canvas's TransformSystem to correct for the current 
        canvas size, framebuffer, and viewport.
        """
        if len(self._fb_stack) == 0:
            fb_size = fb_rect = None
        else:
            fb, origin, fb_size = self._fb_stack[-1]
            fb_rect = origin + fb_size
            
        if len(self._vp_stack) == 0:
            viewport = None
        else:
            viewport = self._vp_stack[-1]
        
        self.transforms.configure(viewport=viewport, fbo_size=fb_size,
                                  fbo_rect=fb_rect)