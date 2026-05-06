def search(self, *args, **kwargs):
        """match all records that have any args in any key/field that also match
        key/value requirements specified in kwargs"""
        ret = []
        for record in Handler.ALL_VERS_DATA.values():
            matchArgs = list(kwargs.keys())
            for k,v in iteritems(kwargs): # restrict records based on key-value match requirement
                try:
                    if record[k] != v: break # a non-matching requirement means this record doesn't match
                except: break # record doesn't have required key 'k'
                matchArgs.remove(k)
            if matchArgs: continue # didn't match all required kwargs
            matchArgs = list(args)
            for k,v in iteritems(record): # find any record with a <value> in it
                if k in matchArgs: matchArgs.remove(k)
                if v in matchArgs: matchArgs.remove(v)
            if matchArgs: continue # didn't match all required args
            ret.append(record)
        return ret