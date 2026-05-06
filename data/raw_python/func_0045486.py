def __extractProperties(self):
        """
        2015-06-04: removed sparql 1.1 queries
        2015-06-03: analogous to get classes

        # instantiate properties making sure duplicates are pruned
        # but the most specific rdftype is kept
        # eg OWL:ObjectProperty over RDF:property

        """
        self.properties = [] # @todo: keep adding?
        self.annotationProperties = []
        self.objectProperties = []
        self.datatypeProperties = []

        qres = self.queryHelper.getAllProperties()

        for candidate in qres:

            test_existing_prop = self.getProperty(uri=candidate[0])
            if not test_existing_prop:
                # create it
                self.properties += [OntoProperty(candidate[0], candidate[1], self.namespaces)]
            else:
                # update it
                if candidate[1] and (test_existing_prop.rdftype == rdflib.RDF.Property):
                    test_existing_prop.rdftype = inferMainPropertyType(candidate[1])


        #add more data
        for aProp in self.properties:

            if aProp.rdftype == rdflib.OWL.DatatypeProperty:
                self.datatypeProperties += [aProp]
            elif aProp.rdftype == rdflib.OWL.AnnotationProperty:
                self.annotationProperties += [aProp]
            elif aProp.rdftype == rdflib.OWL.ObjectProperty:
                self.objectProperties += [aProp]
            else:
                pass

            aProp.triples = self.queryHelper.entityTriples(aProp.uri)
            aProp._buildGraph() # force construction of mini graph

            # attach to an ontology [2015-06-15: no property type distinction yet]
            for uri in aProp.getValuesForProperty(rdflib.RDFS.isDefinedBy):
                onto = self.getOntology(str(uri))
                if onto:
                    onto.properties += [aProp]
                    aProp.ontology = onto



            self.__buildDomainRanges(aProp)

            # add direct Supers
            directSupers = self.queryHelper.getPropDirectSupers(aProp.uri)

            for x in directSupers:
                superprop = self.getProperty(uri=x[0])
                if superprop:
                    aProp._parents.append(superprop)

                    # add inverse relationships (= direct subs for superprop)
                    if aProp not in superprop.children():
                         superprop._children.append(aProp)