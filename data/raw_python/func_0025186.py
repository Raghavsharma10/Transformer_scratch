def max_size(self):
        """
        Gets the largest size of the object over all timesteps.
        
        Returns:
            Maximum size of the object in pixels
        """
        sizes = np.array([m.sum() for m in self.masks])
        return sizes.max()