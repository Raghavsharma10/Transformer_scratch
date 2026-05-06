def clear(self, context=None):
        """ Delete all data from the graph. """
        context = URIRef(context).n3() if context is not None else '?g'
        query = """
            DELETE { GRAPH %s { ?s ?p ?o } } WHERE { GRAPH %s { ?s ?p ?o } }
        """ % (context, context)
        self.parent.graph.update(query)