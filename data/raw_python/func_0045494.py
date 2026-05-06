def getOntology(self, id=None, uri=None, match=None):
        """
        get the saved-ontology with given ID or via other methods...
        """

        if not id and not uri and not match:
            return None

        if type(id) == type("string"):
            uri = id
            id = None
            if not uri.startswith("http://"):
                match = uri
                uri = None
        if match:
            if type(match) != type("string"):
                return []
            res = []
            for x in self.ontologies:
                if match.lower() in x.uri.lower():
                    res += [x]
            return res
        else:
            for x in self.ontologies:
                if id and x.id == id:
                    return x
                if uri and x.uri.lower() == uri.lower():
                    return x
            return None