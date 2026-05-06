def update_child(self, child):
        """ Update child """

        rest_name = child.rest_name
        children = self.fetcher_for_rest_name(rest_name)

        index = None

        for local_child in children:
            if local_child.id == child.id:
                index = children.index(local_child)
                break

        if index is not None:
            children[index] = child