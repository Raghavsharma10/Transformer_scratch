def JSONEncoder_newdefault(kind=['uuid', 'datetime', 'time', 'decimal']):
    '''
    JSON Encoder newdfeault is a wrapper capable of encoding several kinds
    Usage:
        from codenerix.helpers import JSONEncoder_newdefault
        JSONEncoder_newdefault()
    '''
    JSONEncoder_olddefault = json.JSONEncoder.default

    def JSONEncoder_wrapped(self, o):
        '''
        json.JSONEncoder.default = JSONEncoder_newdefault
        '''
        if ('uuid' in kind) and isinstance(o, UUID):
            return str(o)
        if ('datetime' in kind) and isinstance(o, datetime):
            return str(o)
        if ('time' in kind) and isinstance(o, time.struct_time):
            return datetime.fromtimestamp(time.mktime(o))
        if ('decimal' in kind) and isinstance(o, decimal.Decimal):
            return str(o)
        return JSONEncoder_olddefault(self, o)
    json.JSONEncoder.default = JSONEncoder_wrapped