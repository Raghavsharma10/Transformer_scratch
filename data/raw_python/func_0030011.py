def python_type(self):
        """Return the python type for the row, possibly getting it from a valuetype reference """

        from ambry.valuetype import resolve_value_type

        if self.valuetype and resolve_value_type(self.valuetype):
            return resolve_value_type(self.valuetype)._pythontype

        elif self.datatype:
            try:
                return self.types[self.datatype][1]
            except KeyError:
                return resolve_value_type(self.datatype)._pythontype

        else:
            from ambry.exc import ConfigurationError
            raise ConfigurationError("Can't get python_type: neither datatype of valuetype is defined")