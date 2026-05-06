def __extractOntologies(self, exclude_BNodes = False, return_string=False):
        """
        returns Ontology class instances

        [ a owl:Ontology ;
            vann:preferredNamespacePrefix "bsym" ;
            vann:preferredNamespaceUri "http://bsym.bloomberg.com/sym/" ],
        """
        out = []

        qres = self.queryHelper.getOntology()

        if qres:
            # NOTE: SPARQL returns a list of rdflib.query.ResultRow (~ tuples..)

            for candidate in qres:
                if isBlankNode(candidate[0]):
                    if exclude_BNodes:
                        continue
                    else:
                        checkDC_ID = [x for x in self.rdfgraph.objects(candidate[0], rdflib.namespace.DC.identifier)]
                        if checkDC_ID:
                            out += [Ontology(checkDC_ID[0], namespaces=self.namespaces),]
                        else:
                            vannprop = rdflib.URIRef("http://purl.org/vocab/vann/preferredNamespaceUri")
                            vannpref = rdflib.URIRef("http://purl.org/vocab/vann/preferredNamespacePrefix")
                            checkDC_ID = [x for x in self.rdfgraph.objects(candidate[0], vannprop)]
                            if checkDC_ID:
                                checkDC_prefix = [x for x in self.rdfgraph.objects(candidate[0], vannpref)]
                                if checkDC_prefix:
                                    out += [Ontology(checkDC_ID[0],
                                                     namespaces=self.namespaces,
                                                     prefPrefix=checkDC_prefix[0])]
                                else:
                                    out += [Ontology(checkDC_ID[0], namespaces=self.namespaces)]

                else:
                    out += [Ontology(candidate[0], namespaces=self.namespaces)]


        else:
            pass
            # printDebug("No owl:Ontologies found")

        #finally... add all annotations/triples
        self.ontologies = out
        for onto in self.ontologies:
            onto.triples = self.queryHelper.entityTriples(onto.uri)
            onto._buildGraph()