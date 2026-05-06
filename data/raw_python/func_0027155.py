def remove_child(self, child):
        """ Remove a child """

        rest_name = child.rest_name
        children = self.fetcher_for_rest_name(rest_name)

        target_child = None

        for local_child in children:
            if local_child.id == child.id:
                target_child = local_child
                break

        if target_child:
            target_child.parent_object = None
            children.remove(target_child)