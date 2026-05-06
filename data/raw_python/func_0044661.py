def bulk_update(self, named_graph, graph, size, is_add=True):
        """
        Bulk adds or deletes. Triples are chunked into n size groups before
        sending to API. This prevents the API endpoint from timing out.
        """
        context = URIRef(named_graph)
        total = len(graph)
        if total > 0:
            for set_size, nt in self.nt_yielder(graph, size):
                if is_add is True:
                    logger.info("Adding {} statements to <{}>.".format(set_size, named_graph))
                    self.update(u'INSERT DATA { GRAPH %s { %s } }' % (context.n3(), nt.decode('utf-8')))
                else:
                    logger.info("Removing {} statements from <{}>.".format(set_size, named_graph))
                    self.update(u'DELETE DATA { GRAPH %s { %s } }' % (context.n3(), nt.decode('utf-8')))
        return total