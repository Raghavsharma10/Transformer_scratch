def parameter_to_field(self, name):
        """
        Promotes a parameter to a field by creating a new array of same
        size as the other existing fields, filling it with the current
        value of the parameter, and then removing that parameter.
        """
        if name not in self._parameters:
            raise ValueError("no '%s' parameter found" % (name))
        if self._fields.count(name) > 0:
            raise ValueError("field with name '%s' already exists" % (name))

        data = np.array([self._parameters[name]]*self._num_fix)

        self.rm_parameter(name)
        self.add_field(name, data)