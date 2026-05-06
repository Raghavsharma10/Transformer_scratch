def jsonstreamer(func):
    """JSON streamer decorator"""
    def wrapper (self, *args, **kwds):
        gen  = func (self, *args, **kwds)
        yield "["
        firstItem  = True
        for item in gen:
            if not firstItem:
                yield ","
            else:
                firstItem = False
            yield cjson.encode(item)
        yield "]"
    return wrapper