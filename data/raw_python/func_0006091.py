def clone_and_update(self, **kwargs):
        """Clones the object and updates the clone with the args

        @param kwargs: Keyword arguments to set
        @return: The cloned copy with updated values
        """
        cloned = self.clone()
        cloned.update(**kwargs)
        return cloned