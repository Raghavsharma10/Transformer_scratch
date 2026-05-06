def surface(self, tag):
        ''' Get surface string that is associated with a tag object '''
        if tag.cfrom is not None and tag.cto is not None and tag.cfrom >= 0 and tag.cto >= 0:
            return self.text[tag.cfrom:tag.cto]
        else:
            return ''