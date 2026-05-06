def __extractSkosConcepts(self):
        """
        2015-08-19: first draft
        """
        self.skosConcepts = [] # @todo: keep adding?

        qres = self.queryHelper.getSKOSInstances()

        for candidate in qres:

            test_existing_cl = self.getSkosConcept(uri=candidate[0])
            if not test_existing_cl:
                # create it
                self.skosConcepts += [OntoSKOSConcept(candidate[0], None, self.namespaces)]
            else:
                pass

        #add more data
        skos = rdflib.Namespace('http://www.w3.org/2004/02/skos/core#')

        for aConcept in self.skosConcepts:

            aConcept.rdftype = skos['Concept']
            aConcept.triples = self.queryHelper.entityTriples(aConcept.uri)
            aConcept._buildGraph() # force construction of mini graph

            aConcept.queryHelper = self.queryHelper

            # attach to an ontology
            for uri in aConcept.getValuesForProperty(rdflib.RDFS.isDefinedBy):
                onto = self.getOntology(str(uri))
                if onto:
                    onto.skosConcepts += [aConcept]
                    aConcept.ontology = onto

            # add direct Supers
            directSupers = self.queryHelper.getSKOSDirectSupers(aConcept.uri)

            for x in directSupers:
                superclass = self.getSkosConcept(uri=x[0])
                if superclass:
                    aConcept._parents.append(superclass)

                    # add inverse relationships (= direct subs for superclass)
                    if aConcept not in superclass.children():
                         superclass._children.append(aConcept)