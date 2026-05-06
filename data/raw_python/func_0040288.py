def visit(self, node):
        '''The main visit function. Visits the passed-in node and calls
        finalize.
        '''
        for token in self.itervisit(node):
            pass
        result = self.finalize()
        if result is not self:
            return result