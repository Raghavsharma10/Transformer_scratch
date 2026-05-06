def iter(self, root=None):
        """
        Create an iterator of (directory, control_dict) tuples for all valid
        parameter choices in this :class:`Nest`.

        :param root: Root directory
        :rtype: Generator of ``(directory, control_dictionary)`` tuples.
        """
        if root is None:
            return iter(self._controls)
        return ((os.path.join(root, outdir), control)
                for outdir, control in self._controls)