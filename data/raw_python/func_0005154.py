def pretty(self, obj=None, display=True):
        """ Formats @obj or :prop:obj

            @obj: the object you'd like to prettify

            -> #str pretty object
        """
        ret = self._format_obj(obj if obj is not None else self.obj)
        if display:
            print(ret)
        else:
            return ret