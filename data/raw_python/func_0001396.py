def new_tag(self, label, cfrom=-1, cto=-1, tagtype='', **kwargs):
        ''' Create a sentence-level tag '''
        tag_obj = Tag(label, cfrom, cto, tagtype=tagtype, **kwargs)
        return self.add_tag(tag_obj)