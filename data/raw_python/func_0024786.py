def from_parameter(self, parameter):
        """Set internal raw state from parameter."""
        if not isinstance(parameter, Parameter):
            raise Exception("parameter::from_parameter_wrong_object")
        self.raw = parameter.raw