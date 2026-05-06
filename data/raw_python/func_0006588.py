def get_layout(self, name):
        '''
        Returns the layout with the given name
        '''
        layout_chain = []

        # Retrieve initial layout file
        try:
            json_data = self.json_files[self.layout_names[name]]
        except KeyError:
            log.error('Could not find layout: %s', name)
            log.error('Layouts path: %s', self.layout_path)
            raise
        layout_chain.append(Layout(name, json_data))

        # Recursively locate parent layout files
        parent = layout_chain[-1].parent()
        while parent is not None:
            # Find the parent
            parent_path = None
            for path in self.json_file_paths:
                if os.path.normcase(os.path.normpath(parent)) in os.path.normcase(path):
                    parent_path = path

            # Make sure a path was found
            if parent_path is None:
                raise UnknownLayoutPathException('Could not find: {}'.format(parent_path))

            # Build layout for parent
            json_data = self.json_files[parent_path]
            layout_chain.append(Layout(parent_path, json_data))

            # Check parent of parent
            parent = layout_chain[-1].parent()

        # Squash layout files
        layout = self.squash_layouts(layout_chain)
        return layout