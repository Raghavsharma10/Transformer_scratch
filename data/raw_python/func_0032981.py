def query(self, parents=None):
        """ Compose the query and generate SPARQL. """
        # TODO: benchmark single-query strategy
        q = Select([])
        q = self.project(q, parent=True)
        q = self.filter(q, parents=parents)

        if self.parent is None:
            subq = Select([self.var])
            subq = self.filter(subq, parents=parents)
            subq = subq.offset(self.node.offset)
            subq = subq.limit(self.node.limit)
            subq = subq.distinct()
            # TODO: sorting.
            subq = subq.order_by(desc(self.var))
            q = q.where(subq)

        # if hasattr(self.context, 'identifier'):
        #     q._where = graph(self.context.identifier, q._where)
        log.debug("Compiled query: %r", q.compile())
        return q