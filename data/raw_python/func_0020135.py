def meta(self, meta):
        '''Extract model metadata for lua script stdnet/lib/lua/odm.lua'''
        data = meta.as_dict()
        data['namespace'] = self.basekey(meta)
        return data