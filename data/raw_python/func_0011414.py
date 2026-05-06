def _pfp__handle_non_consecutive_duplicate(self, name, child, insert=True):
        """This new child, and potentially one already existing child, need to
        have a numeric suffix appended to their name.
        
        An entry will be made for this name in ``self._pfp__name_collisions`` to keep
        track of the next available suffix number"""
        if name in self._pfp__children_map:
            previous_child = self._pfp__children_map[name]

            # DO NOT cause __eq__ to be called, we want to test actual objects, not comparison
            # operators
            if previous_child is not child:
                self._pfp__handle_non_consecutive_duplicate(name, previous_child, insert=False)
                del self._pfp__children_map[name]
        
        next_suffix = self._pfp__name_collisions.setdefault(name, 0)
        new_name = "{}_{}".format(name, next_suffix)
        child._pfp__name = new_name
        self._pfp__name_collisions[name] = next_suffix + 1
        self._pfp__children_map[new_name] = child
        child._pfp__parent = self

        if insert:
            self._pfp__children.append(child)

        return child