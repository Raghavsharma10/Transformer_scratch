def valuetype_class(self):
        """Return the valuetype class, if one is defined, or a built-in type if it isn't"""

        from ambry.valuetype import resolve_value_type

        if self.valuetype:
            return resolve_value_type(self.valuetype)

        else:
            return resolve_value_type(self.datatype)