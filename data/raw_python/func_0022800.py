def get_scene_bounds(self, dim=None):
        """Get the total bounds based on the visuals present in the scene

        Parameters
        ----------
        dim : int | None
            Dimension to return.

        Returns
        -------
        bounds : list | tuple
            If ``dim is None``, Returns a list of 3 tuples, otherwise
            the bounds for the requested dimension.
        """
        # todo: handle sub-children
        # todo: handle transformations
        # Init
        bounds = [(np.inf, -np.inf), (np.inf, -np.inf), (np.inf, -np.inf)]
        # Get bounds of all children
        for ob in self.scene.children:
            if hasattr(ob, 'bounds'):
                for axis in (0, 1, 2):
                    if (dim is not None) and dim != axis:
                        continue
                    b = ob.bounds(axis)
                    if b is not None:
                        b = min(b), max(b)  # Ensure correct order
                        bounds[axis] = (min(bounds[axis][0], b[0]), 
                                        max(bounds[axis][1], b[1]))
        # Set defaults
        for axis in (0, 1, 2):
            if any(np.isinf(bounds[axis])):
                bounds[axis] = -1, 1
        
        if dim is not None:
            return bounds[dim]
        else:
            return bounds