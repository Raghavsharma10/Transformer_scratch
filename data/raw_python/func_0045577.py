def getAllClasses(self, hide_base_schemas=True):
        """
        by default, obscure all RDF/RDFS/OWL/XML stuff
        2016-05-06: not obscured anymore
        """
        query = """SELECT DISTINCT ?x ?c
                 WHERE {
                         {
                             { ?x a owl:Class }
                             union
                             { ?x a rdfs:Class }
                             union
                             { ?x rdfs:subClassOf ?y }
                             union
                             { ?z rdfs:subClassOf ?x }
                             union
                             { ?y rdfs:domain ?x }
                             union
                             { ?y rdfs:range ?x }
                             # union
                             # { ?y rdf:type ?x }
                         } .

                         ?x a ?c

                    %s

                 }
                 ORDER BY  ?x
                 """
        if hide_base_schemas:
            query = query %  """FILTER(
                     !STRSTARTS(STR(?x), "http://www.w3.org/2002/07/owl")
                     && !STRSTARTS(STR(?x), "http://www.w3.org/1999/02/22-rdf-syntax-ns")
                     && !STRSTARTS(STR(?x), "http://www.w3.org/2000/01/rdf-schema")
                     && !STRSTARTS(STR(?x), "http://www.w3.org/2001/XMLSchema")
                     && !STRSTARTS(STR(?x), "http://www.w3.org/XML/1998/namespace")
                     && (!isBlank(?x))
                      ) ."""
        else:
            query = query % ""

        qres = self.rdfgraph.query(query)
        return list(qres)