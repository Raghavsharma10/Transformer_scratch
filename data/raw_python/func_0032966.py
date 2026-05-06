def delete(self):
        """ Delete all statements matching the current context identifier
        from the main store. """
        if self.parent.buffered:
            query = 'CLEAR SILENT GRAPH %s ;' % self.identifier.n3()
            self.parent.graph.update(query)
            self.flush()
        else:
            self.graph.remove((None, None, None))