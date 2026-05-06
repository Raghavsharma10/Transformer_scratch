def __extractClasses(self, hide_base_schemas=True):
        """
        2015-06-04: removed sparql 1.1 queries
        2015-05-25: optimized via sparql queries in order to remove BNodes
        2015-05-09: new attempt

        Note: queryHelper.getAllClasses() returns a list of tuples,
        (class, classRDFtype)
        so in some cases there are duplicates if a class is both RDFS.CLass and OWL.Class
        In this case we keep only OWL.Class as it is more informative.
        """


        self.classes = [] # @todo: keep adding?

        qres = self.queryHelper.getAllClasses(hide_base_schemas=hide_base_schemas)

        for class_tuple in qres:
            
            _uri = class_tuple[0]
            try:
                _type = class_tuple[1]
            except:
                _type= ""

            test_existing_cl = self.getClass(uri=_uri)
            if not test_existing_cl:
                # create it
                self.classes += [OntoClass(_uri, _type, self.namespaces)]
            else:
                # if OWL.Class over RDFS.Class - update it
                if _type == rdflib.OWL.Class:
                    test_existing_cl.rdftype = rdflib.OWL.Class



        #add more data
        for aClass in self.classes:

            aClass.triples = self.queryHelper.entityTriples(aClass.uri)
            aClass._buildGraph() # force construction of mini graph

            aClass.queryHelper = self.queryHelper

            # attach to an ontology
            for uri in aClass.getValuesForProperty(rdflib.RDFS.isDefinedBy):
                onto = self.getOntology(uri)
                if onto:
                    onto.classes += [aClass]
                    aClass.ontology = onto

            # add direct Supers
            directSupers = self.queryHelper.getClassDirectSupers(aClass.uri)

            for x in directSupers:
                superclass = self.getClass(uri=x[0])
                if superclass:
                    aClass._parents.append(superclass)

                    # add inverse relationships (= direct subs for superclass)
                    if aClass not in superclass.children():
                         superclass._children.append(aClass)