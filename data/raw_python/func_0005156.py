def pretty_objname(self, obj=None, maxlen=50, color="boldcyan"):
        """ Pretty prints object name

            @obj: the object whose name you want to pretty print
            @maxlen: #int maximum length of an object name to print
            @color: your choice of :mod:colors or |None|

            -> #str pretty object name
            ..
                from vital.debug import Look
                print(Look.pretty_objname(dict))
                # -> 'dict\x1b[1;36m<builtins>\x1b[1;m'
            ..
        """
        parent_name = lambda_sub("", get_parent_name(obj) or "")
        objname = get_obj_name(obj)
        if color:
            objname += colorize("<{}>".format(parent_name), color, close=False)
        else:
            objname += "<{}>".format(parent_name)
        objname = objname if len(objname) < maxlen else \
            objname[:(maxlen-1)]+"…>"
        if color:
            objname += colors.RESET
        return objname