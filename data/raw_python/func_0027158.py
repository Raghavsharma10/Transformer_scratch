def from_dict(self, dictionary):
        """ Sets all the exposed ReST attribues from the given dictionary

            Args:
                dictionary (dict): dictionnary containing the raw object attributes and their values.

            Example:
                >>> info = {"name": "my group", "private": False}
                >>> group = NUGroup()
                >>> group.from_dict(info)
                >>> print "name: %s - private: %s" % (group.name, group.private)
                "name: my group - private: False"
        """

        for remote_name, remote_value in dictionary.items():
            # Check if a local attribute is exposed with the remote_name
            # if no attribute is exposed, return None
            local_name = next((name for name, attribute in self._attributes.items() if attribute.remote_name == remote_name), None)

            if local_name:
                setattr(self, local_name, remote_value)
            else:
                # print('Attribute %s could not be added to object %s' % (remote_name, self))
                pass