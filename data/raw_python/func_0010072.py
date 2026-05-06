def define_attribute(self, name, atype, data=None):
        """
        Define a new attribute. atype has to be one of 'integer', 'real', 'numeric', 'string', 'date' or 'nominal'.
        For nominal attributes, pass the possible values as data.
        For date attributes, pass the format as data.
        """
        self.attributes.append(name)
        assert atype in TYPES, "Unknown type '%s'. Must be one of: %s" % (atype, ', '.join(TYPES),)
        self.attribute_types[name] = atype
        self.attribute_data[name] = data