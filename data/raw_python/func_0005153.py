def objname(self, obj=None):
        """ Formats object names in a pretty fashion """
        obj = obj or self.obj
        _objname = self.pretty_objname(obj, color=None)
        _objname = "'{}'".format(colorize(_objname, "blue"))
        return _objname