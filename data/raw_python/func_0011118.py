def copy_with_new_str(self, new_str):
        """Copies the current FmtStr's attributes while changing its string."""
        # What to do when there are multiple Chunks with conflicting atts?
        old_atts = dict((att, value) for bfs in self.chunks
                    for (att, value) in bfs.atts.items())
        return FmtStr(Chunk(new_str, old_atts))