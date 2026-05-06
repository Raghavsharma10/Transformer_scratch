def tofile(self, filepath=None):
        """Saves configuration into a file and returns its path.

        Convenience method.

        :param str|unicode filepath: Filepath to save configuration into.
            If not provided a temporary file will be automatically generated.

        :rtype: str|unicode

        """
        if filepath is None:
            with NamedTemporaryFile(prefix='%s_' % self.alias, suffix='.ini', delete=False) as f:
                filepath = f.name

        else:
            filepath = os.path.abspath(filepath)

            if os.path.isdir(filepath):
                filepath = os.path.join(filepath, '%s.ini' % self.alias)

        with open(filepath, 'w') as target_file:
            target_file.write(self.format())
            target_file.flush()

        return filepath