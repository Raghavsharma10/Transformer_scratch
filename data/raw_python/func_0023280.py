def configure(self, viewport=None, fbo_size=None, fbo_rect=None,
                  canvas=None):
        """Automatically configure the TransformSystem:

        * canvas_transform maps from the Canvas logical pixel
          coordinate system to the framebuffer coordinate system, taking into 
          account the logical/physical pixel scale factor, current FBO 
          position, and y-axis inversion.
        * framebuffer_transform maps from the current GL viewport on the
          framebuffer coordinate system to clip coordinates (-1 to 1). 
          
          
        Parameters
        ==========
        viewport : tuple or None
            The GL viewport rectangle (x, y, w, h). If None, then it
            is assumed to cover the entire canvas.
        fbo_size : tuple or None
            The size of the active FBO. If None, then it is assumed to have the
            same size as the canvas's framebuffer.
        fbo_rect : tuple or None
            The position and size (x, y, w, h) of the FBO in the coordinate
            system of the canvas's framebuffer. If None, then the bounds are
            assumed to cover the entire active framebuffer.
        canvas : Canvas instance
            Optionally set the canvas for this TransformSystem. See the 
            `canvas` property.
        """
        # TODO: check that d2f and f2r transforms still contain a single
        # STTransform (if the user has modified these, then auto-config should
        # either fail or replace the transforms)
        if canvas is not None:
            self.canvas = canvas
        canvas = self._canvas
        if canvas is None:
            raise RuntimeError("No canvas assigned to this TransformSystem.")
       
        # By default, this should invert the y axis--canvas origin is in top
        # left, whereas framebuffer origin is in bottom left.
        map_from = [(0, 0), canvas.size]
        map_to = [(0, canvas.physical_size[1]), (canvas.physical_size[0], 0)]
        self._canvas_transform.transforms[1].set_mapping(map_from, map_to)

        if fbo_rect is None:
            self._canvas_transform.transforms[0].scale = (1, 1, 1)
            self._canvas_transform.transforms[0].translate = (0, 0, 0)
        else:
            # Map into FBO coordinates
            map_from = [(fbo_rect[0], fbo_rect[1]),
                        (fbo_rect[0] + fbo_rect[2], fbo_rect[1] + fbo_rect[3])]
            map_to = [(0, 0), fbo_size]
            self._canvas_transform.transforms[0].set_mapping(map_from,  map_to)
            
        if viewport is None:
            if fbo_size is None:
                # viewport covers entire canvas
                map_from = [(0, 0), canvas.physical_size]
            else:
                # viewport covers entire FBO
                map_from = [(0, 0), fbo_size]
        else:
            map_from = [viewport[:2], 
                        (viewport[0] + viewport[2], viewport[1] + viewport[3])]
        map_to = [(-1, -1), (1, 1)]
        self._framebuffer_transform.transforms[0].set_mapping(map_from, map_to)