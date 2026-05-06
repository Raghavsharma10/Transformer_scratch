def fromInputs(self, inputs):
        """
        Extract the inputs associated with this parameter from the given
        dictionary and coerce them using C{self.coercer}.

        @type inputs: C{dict} mapping C{str} to C{list} of C{str}
        @param inputs: The contents of a form post, in the conventional
            structure.

        @rtype: L{Deferred}
        @return: A Deferred which will be called back with a list of the
            structured data associated with this parameter.
        """
        outputs = []
        for i in xrange(self.count):
            name = self.name + '_' + str(i)
            try:
                value = inputs[name][0]
            except KeyError:
                raise ConfigurationError(
                    "Missing value for field %d of %s" % (i, self.name))
            else:
                outputs.append(maybeDeferred(self.coercer, value))
        return gatherResults(outputs)