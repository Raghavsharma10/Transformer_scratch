def copy_with_new_atts(self, **attributes):
        """Returns a new FmtStr with the same content but new formatting"""
        return FmtStr(*[Chunk(bfs.s, bfs.atts.extend(attributes))
                        for bfs in self.chunks])