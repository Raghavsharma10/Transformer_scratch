def set_transform(self, type_, *args, **kwargs):
        """ Create a new transform of *type* and assign it to this node.

        All extra arguments are used in the construction of the transform.

        Parameters
        ----------
        type_ : str
            The transform type.
        *args : tuple
            Arguments.
        **kwargs : dict
            Keywoard arguments.
        """
        self.transform = create_transform(type_, *args, **kwargs)