def filter(self, q, parents=None):
        """ Apply any filters to the query. """
        if self.node.leaf and self.node.filtered:
            # TODO: subject filters?
            q = q.where((self.parent.var,
                         self.predicate,
                         self.var))
            # TODO: inverted nodes
            q = self.filter_value(q, self.var)
        elif self.parent is not None:
            q = q.where((self.parent.var, self.predicate, self.var))

            if parents is not None:
                parents = [URIRef(p) for p in parents]
                q = q.filter(self.parent.var.in_(*parents))

        # TODO: forbidden nodes.
        for child in self.children:
            q = child.filter(q)
        return q