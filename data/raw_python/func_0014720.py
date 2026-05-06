def append(self, tag):
        '''
            append - Append an item to this tag collection

            @param tag - an AdvancedTag
        '''
        list.append(self, tag)
        self.uids.add(tag.uid)