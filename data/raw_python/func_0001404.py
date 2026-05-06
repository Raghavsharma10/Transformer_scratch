def tag_map(self):
        ''' Build a map from tagtype to list of tags '''
        tm = dd(list)
        for tag in self.__tags:
            tm[tag.tagtype].append(tag)
        return tm