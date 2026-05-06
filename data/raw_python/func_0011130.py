def new_with_atts_removed(self, *attributes):
        """Returns a new FmtStr with the same content but some attributes removed"""
        return FmtStr(*[Chunk(bfs.s, bfs.atts.remove(*attributes))
                        for bfs in self.chunks])