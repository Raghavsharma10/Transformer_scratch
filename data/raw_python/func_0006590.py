def squash_layouts(self, layouts):
        '''
        Returns a squashed layout

        The first element takes precedence (i.e. left to right).
        Dictionaries are recursively merged, overwrites only occur on non-dictionary entries.

        [0,1]

        0:
        test: 'my data'

        1:
        test: 'stuff'

        Result:
        test: 'my data'

        @param layouts: List of layouts to merge together
        @return: New layout with list of layouts squash merged
        '''
        top_layout = layouts[0]
        json_data = {}

        # Generate a new container Layout
        layout = Layout(top_layout.name(), json_data, layouts)

        # Merge in each of the layouts
        for mlayout in reversed(layouts):
            # Overwrite all fields, *except* dictionaries
            # For dictionaries, keep recursing until non-dictionaries are found
            self.dict_merge(layout.json(), mlayout.json())

        return layout