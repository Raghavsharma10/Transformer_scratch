def _bindingsToDict(self, bindings):
        """
        Given a binding from the sparql query result,
        create a dict of plain text
        """
        myDict = {}
        for key, val in bindings.iteritems():
            myDict[key.toPython().replace('?', '')] = val.toPython()
        return myDict