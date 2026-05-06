def add_property(self, set_property, name, starting_value, tag_name=None):
        """ Set properies of atributes stored in content using stored common fdel and fget and given fset.

            Args:
                set_property -- Function that sets given property.
                name -- Name of the atribute this property must simulate. Used as key in content dict by default.
                starting_value -- Starting value of given property.

            Keyword args:
                tag_name -- The tag name stored in conted dict as a key if different to name.
        """
        def del_property(self, tag_name):
            try:
                del self._content[tag_name]
            except KeyError:
                pass

        def get_property(self, tag_name):
            try:
                return self._content[tag_name]
            except KeyError:
                return None

        tag_name = (name if tag_name is None else tag_name)
        fget = lambda self: get_property(self, tag_name)
        fdel = lambda self: del_property(self, tag_name)
        fset = lambda self, value: set_property(value)
        setattr(self.__class__, name, property(fget, fset, fdel))
        set_property(starting_value)