def getEntity(self, id=None, uri=None, match=None):
        """
        get a generic entity with given ID or via other methods...
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
            if ":" in match: # qname
                for x in self.classes:
                    if match.lower() in x.qname.lower():
                        res += [x]
                for x in self.properties:
                    if match.lower() in x.qname.lower():
                        res += [x]
            else:
                for x in self.classes:
                    if match.lower() in x.uri.lower():
                        res += [x]
                for x in self.properties:
                    if match.lower() in x.uri.lower():
                        res += [x]
            return res
        else:
            for x in self.classes:
                if id and x.id == id:
                    return x
                if uri and x.uri.lower() == uri.lower():
                    return x
            for x in self.properties:
                if id and x.id == id:
                    return x
                if uri and x.uri.lower() == uri.lower():
                    return x
            return None