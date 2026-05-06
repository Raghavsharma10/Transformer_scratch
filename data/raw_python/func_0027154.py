def add_child(self, child):
        """ Add a child """

        rest_name = child.rest_name
        children = self.fetcher_for_rest_name(rest_name)

        if children is None:
            raise InternalConsitencyError('Could not find fetcher with name %s while adding %s in parent %s' % (rest_name, child, self))

        if child not in children:
            child.parent_object = self
            children.append(child)