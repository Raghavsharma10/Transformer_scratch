def merge_uris(self, uri1, uri2, graph=DEFAULT_GRAPH):
        """
        Generate statements to merge two URIS in a specified graph.
        """
        rq = """
        CONSTRUCT {
            ?uri ?p ?o .
            ?other ?p2 ?uri .
        }
        WHERE {
          GRAPH ?g {
            {
                ?uri ?p ?o
            } UNION {
                ?other ?p2 ?uri
            }
          }
        }
        """
        for var in [uri1, uri2, graph]:
            assert type(var) == URIRef
        addg = Graph()
        removeg = Graph()
        rsp2 = self.query(rq, initBindings=dict(uri=uri2, g=graph))

        # reassign triples were the merged uri is the subject
        for pred, obj in rsp2.graph.predicate_objects(subject=uri2):
            addg.add((uri1, pred, obj))
            removeg.add((uri2, pred, obj))

        # reassign triples were the merged uri is the object
        for subj, pred in rsp2.graph.subject_predicates(object=uri2):
            addg.add((subj, pred, uri1))
            removeg.add((subj, pred, uri2))

        return addg, removeg