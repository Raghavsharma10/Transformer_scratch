def set_perspective(self, fov, aspect, near, far):
        """Set the perspective

        Parameters
        ----------
        fov : float
            Field of view.
        aspect : float
            Aspect ratio.
        near : float
            Near location.
        far : float
            Far location.
        """
        self.matrix = transforms.perspective(fov, aspect, near, far)