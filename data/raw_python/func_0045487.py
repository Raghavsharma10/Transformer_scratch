def __buildDomainRanges(self, aProp):
        """
        extract domain/range details and add to Python objects
        """
        domains = aProp.rdfgraph.objects(None, rdflib.RDFS.domain)
        ranges =  aProp.rdfgraph.objects(None, rdflib.RDFS.range)

        for x in domains:
            if not isBlankNode(x):
                aClass = self.getClass(uri=str(x))
                if aClass:
                    aProp.domains += [aClass]
                    aClass.domain_of += [aProp]
                else:
                    aProp.domains += [x]  # edge case: it's not an OntoClass instance?

        for x in ranges:
            if not isBlankNode(x):
                aClass = self.getClass(uri=str(x))
                if aClass:
                    aProp.ranges += [aClass]
                    aClass.range_of += [aProp]
                else:
                    aProp.ranges += [x]