def fromInputs(self, received):
        """
        Convert some random strings received from a browser into structured
        data, using a list of parameters.

        @param received: a dict of lists of strings, i.e. the canonical Python
            form of web form post.

        @rtype: L{Deferred}
        @return: A Deferred which will be called back with a dict mapping
            parameter names to coerced parameter values.
        """
        results = []
        for parameter in self.parameters:
            name = parameter.name.encode('ascii')
            d = maybeDeferred(parameter.fromInputs, received)
            d.addCallback(lambda value, name=name: (name, value))
            results.append(d)
        return gatherResults(results).addCallback(dict)