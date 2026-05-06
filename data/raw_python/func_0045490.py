def getClass(self, id=None, uri=None, match=None):
        """
        get the saved-class with given ID or via other methods...

        Note: it tries to guess what is being passed..

        In [1]: g.getClass(uri='http://www.w3.org/2000/01/rdf-schema#Resource')
        Out[1]: <Class *http://www.w3.org/2000/01/rdf-schema#Resource*>

        In [2]: g.getClass(10)
        Out[2]: <Class *http://purl.org/ontology/bibo/AcademicArticle*>

        In [3]: g.getClass(match="person")
        Out[3]:
        [<Class *http://purl.org/ontology/bibo/PersonalCommunicationDocument*>,
         <Class *http://purl.org/ontology/bibo/PersonalCommunication*>,
         <Class *http://xmlns.com/foaf/0.1/Person*>]

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
            else:
                for x in self.classes:
                    if match.lower() in x.uri.lower():
                        res += [x]
            return res
        else:
            for x in self.classes:
                if id and x.id == id:
                    return x
                if uri and x.uri.lower() == uri.lower():
                    return x
            return None