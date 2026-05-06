def removeduplicates(self, entries = None):
        '''
        Loop over children a remove duplicate entries.
        @return - a list of removed entries
        '''
        removed = []
        if entries == None:
            entries = {}
        new_children = []
        for c in self.children:
            cs = str(c)
            cp = entries.get(cs,None)
            if cp:
                new_children.append(cp)
                removed.append(c)
            else:
                dups = c.removeduplicates(entries)
                if dups:
                    removed.extend(dups)
                entries[cs] = c
                new_children.append(c)
        self.children = new_children
        return removed