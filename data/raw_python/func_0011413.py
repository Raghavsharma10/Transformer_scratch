def _pfp__add_child(self, name, child, stream=None, overwrite=False):
        """Add a child to the Struct field. If multiple consecutive fields are
        added with the same name, an implicit array will be created to store
        all fields of that name.

        :param str name: The name of the child
        :param pfp.fields.Field child: The field to add
        :param bool overwrite: Overwrite existing fields (False)
        :param pfp.bitwrap.BitwrappedStream stream: unused, but her for compatability with Union._pfp__add_child
        :returns: The resulting field added
        """
        if not overwrite and self._pfp__is_non_consecutive_duplicate(name, child):
            return self._pfp__handle_non_consecutive_duplicate(name, child)
        elif not overwrite and name in self._pfp__children_map:
            return self._pfp__handle_implicit_array(name, child)
        else:
            child._pfp__parent = self
            self._pfp__children.append(child)
            child._pfp__name = name
            self._pfp__children_map[name] = child
            return child