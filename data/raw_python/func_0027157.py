def to_dict(self):
        """ Converts the current object into a Dictionary using all exposed ReST attributes.

            Returns:
                dict: the dictionary containing all the exposed ReST attributes and their values.

            Example::
                >>> print entity.to_dict()
                {"name": "my entity", "description": "Hello World", "ID": "xxxx-xxx-xxxx-xxx", ...}
        """

        dictionary = dict()

        for local_name, attribute in self._attributes.items():
            remote_name = attribute.remote_name

            if hasattr(self, local_name):
                value = getattr(self, local_name)

                # Removed to resolve issue http://mvjira.mv.usa.alcatel.com/browse/VSD-5940 (12/15/2014)
                # if isinstance(value, bool):
                #     value = int(value)

                if isinstance(value, NURESTObject):
                    value = value.to_dict()

                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], NURESTObject):
                    tmp = list()
                    for obj in value:
                        tmp.append(obj.to_dict())

                    value = tmp

                dictionary[remote_name] = value
            else:
                pass  # pragma: no cover

        return dictionary