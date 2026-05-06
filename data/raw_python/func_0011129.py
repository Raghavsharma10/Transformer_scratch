def shared_atts(self):
        """Gets atts shared among all nonzero length component Chunk"""
        #TODO cache this, could get ugly for large FmtStrs
        atts = {}
        first = self.chunks[0]
        for att in sorted(first.atts):
            #TODO how to write this without the '???'?
            if all(fs.atts.get(att, '???') == first.atts[att] for fs in self.chunks if len(fs) > 0):
                atts[att] = first.atts[att]
        return atts