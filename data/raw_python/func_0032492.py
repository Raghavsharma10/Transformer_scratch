def fromInputs(self, inputs):
        """
        Extract the inputs associated with the child forms of this parameter
        from the given dictionary and coerce them using C{self.coercer}.

        @type inputs: C{dict} mapping C{unicode} to C{list} of C{unicode}
        @param inputs: The contents of a form post, in the conventional
            structure.

        @rtype: L{Deferred}
        @return: The structured data associated with this parameter represented
            by the post data.
        """
        try:
            values = inputs[self.name]
        except KeyError:
            raise ConfigurationError(
                "Missing value for input: " + self.name)
        return self.coerceMany(values)