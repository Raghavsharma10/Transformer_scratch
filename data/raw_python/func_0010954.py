def value_type(self):
        """ The attribute's type, note that this is the type of the attribute's
        value and not its affect on the item (i.e. negative or positive). See
        'type' for that. """
        redundantprefix = "value_is_"
        vtype = self._attribute.get("description_format")

        if vtype and vtype.startswith(redundantprefix):
            return vtype[len(redundantprefix):]
        else:
            return vtype