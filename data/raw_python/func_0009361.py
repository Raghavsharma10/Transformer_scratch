def matches(self, verb, params):
        """ Test if the method matches the provided set of arguments

        :param verb: HTTP verb. Uppercase
        :type verb: str
        :param params: Existing route parameters
        :type params: set
        :returns: Whether this view matches
        :rtype: bool
        """
        return (self.ifset   is None or self.ifset          <= params) and \
               (self.ifnset  is None or self.ifnset.isdisjoint(params)) and \
               (self.methods is None or verb in self.methods)