def project(self, q, parent=False):
        """ Figure out which attributes should be returned for the current
        level of the query. """
        if self.parent:
            print (self.parent.var, self.predicate, self.var)
        q = q.project(self.var, append=True)
        if parent and self.parent:
            q = q.project(self.parent.var, append=True)
        if not self.node.specific_attribute:
            q = q.project(self.predicate, append=True)
        for child in self.children:
            if child.node.leaf:
                q = child.project(q)
        return q