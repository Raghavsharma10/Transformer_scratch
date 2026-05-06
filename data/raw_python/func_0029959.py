def field_row(self, fields):
        """
        Return a list of values to match the fields values. This is used when listing bundles to
        produce a table of information about the bundle.

        :param fields: A list of names of data items.
        :return: A list of values, in the same order as the fields input

        The names in the fields llist can be:

        - state: The current build state
        - source_fs: The URL of the build source filesystem
        - about.*: Any of the metadata fields in the about section

        """

        row = self.dataset.row(fields)

        # Modify for special fields
        for i, f in enumerate(fields):
            if f == 'bstate':
                row[i] = self.state
            elif f == 'dstate':
                row[i] = self.dstate
            elif f == 'source_fs':
                row[i] = self.source_fs
            elif f.startswith('about'):  # all metadata in the about section, ie: about.title
                _, key = f.split('.')
                row[i] = self.metadata.about[key]
            elif f.startswith('state'):
                _, key = f.split('.')
                row[i] = self.buildstate.state[key]
            elif f.startswith('count'):
                _, key = f.split('.')
                if key == 'sources':
                    row[i] = len(self.dataset.sources)
                elif key == 'tables':
                    row[i] = len(self.dataset.tables)

        return row