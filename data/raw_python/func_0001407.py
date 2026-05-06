def new_tag(self, label, cfrom=None, cto=None, tagtype=None, **kwargs):
        ''' Create a new tag on this token '''
        if cfrom is None:
            cfrom = self.cfrom
        if cto is None:
            cto = self.cto
        tag = Tag(label=label, cfrom=cfrom, cto=cto, tagtype=tagtype, **kwargs)
        return self.add_tag(tag)