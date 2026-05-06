def genealogic_types(self):
        """ Get genealogic types

            Returns:
                Returns a list of all parent types
        """

        types = []
        parent = self

        while parent:
            types.append(parent.rest_name)
            parent = parent.parent_object

        return types