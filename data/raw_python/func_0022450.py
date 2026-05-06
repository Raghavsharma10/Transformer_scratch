def vec_by_id(self, docid):
        """Return indexed vector corresponding to document `docid`."""
        pos = self.id2pos[docid]
        return self.qindex.vector_by_id(pos)