def getElementById(self, _id):
        '''
            getElementById - Gets an element within this collection by id

            @param _id - string of "id" attribute

            @return - a single tag matching the id, or None if none found
        '''
        for tag in self:
            if tag.id == _id:
                return tag
            for subtag in tag.children:
                tmp = subtag.getElementById(_id)
                if tmp is not None:
                    return tmp
        return None