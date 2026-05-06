def get_content_field(self, name):
        """ Get the contents of a specific subtag from Clusterpoint Storage's response's content tag.

            Args:
                name -- A name string of the content's subtag to be returned.

            Returns:
                A dict representing the contents of the specified field or a list of dicts
                if there are multiple fields with that tag name. Returns None if no field found.
        """
        fields = self._content.findall(name)
        if not fields:
            return None
        elif len(fields) == 1:
            return etree_to_dict(fields[0])[name]
        else:
            return [etree_to_dict(field)[name] for field in fields]