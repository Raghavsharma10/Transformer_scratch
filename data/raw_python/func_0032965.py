def save(self):
        """ Transfer the statements in this context over to the main store. """
        if self.parent.buffered:
            query = """
                INSERT DATA { GRAPH %s { %s } }
            """
            query = query % (self.identifier.n3(),
                             self.graph.serialize(format='nt'))
            self.parent.graph.update(query)
            self.flush()
        else:
            self.meta.generate()