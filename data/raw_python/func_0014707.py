def getElementById(self, _id):
        '''
            getElementById - Search children of this tag for a tag containing an id

            @param _id - String of id

            @return - AdvancedTag or None
        '''
        for child in self.children:
            if child.getAttribute('id') == _id:
                return child
            found = child.getElementById(_id)
            if found is not None:
                return found
        return None