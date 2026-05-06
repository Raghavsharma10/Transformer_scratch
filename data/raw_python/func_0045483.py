def stats(self):
        """ shotcut to pull out useful info for a graph 

        2016-08-18 the try/except is a dirty solution to a problem
        emerging with counting graph lenght on cached Graph objects..
        TODO: investigate what's going on..
        """
        out = []
        try:
            out += [("Triples", len(self.rdfgraph))]
        except:
            pass
        out += [("Classes", len(self.classes))]
        out += [("Properties", len(self.properties))]
        out += [("Annotation Properties", len(self.annotationProperties))]
        out += [("Object Properties", len(self.objectProperties))]
        out += [("Datatype Properties", len(self.datatypeProperties))]
        out += [("Skos Concepts", len(self.skosConcepts))]
        # out += [("Individuals", len(self.instances))]
        return out