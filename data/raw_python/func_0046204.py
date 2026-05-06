def get_as_list(self, tag_name):
        """
        Return the value of a tag, making sure that it's a list.  Absent
        tags are returned as an empty-list; single tags are returned as a
        one-element list.

        The returned list is a copy, and modifications do not affect the
        original object.
        """
        val = self.get(tag_name, [])
        if isinstance(val, list):
            return val[:]
        else:
            return [val]