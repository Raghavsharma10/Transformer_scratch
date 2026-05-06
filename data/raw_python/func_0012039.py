def update(xCqNck7t, **kwargs):
        """Updates the Dict with the given values. Turns internal dicts into Dicts."""
        def dict_list_val(inlist):
            l = []
            for i in inlist:
                if type(i)==dict:
                    l.append(Dict(**i))
                elif type(i)==list:
                    l.append(make_list(i))
                elif type(i)==bytes:
                    l.append(i.decode('UTF-8'))
                else:
                    l.append(i)
            return l
        for k in list(kwargs.keys()):
            if type(kwargs[k])==dict:
                xCqNck7t[k] = Dict(**kwargs[k])
            elif type(kwargs[k])==list:
                xCqNck7t[k] = dict_list_val(kwargs[k])
            else:
                xCqNck7t[k] = kwargs[k]