def parent_for_matching_rest_name(self, rest_names):
        """ Return parent that matches a rest name """

        parent = self

        while parent:
            if parent.rest_name in rest_names:
                return parent

            parent = parent.parent_object

        return None