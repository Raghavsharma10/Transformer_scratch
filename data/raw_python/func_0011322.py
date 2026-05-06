def __makeattr(self, name):
        """lazily compute value for name or raise AttributeError if unknown."""
        # print "makeattr", self.__name__, name
        target = None
        if '__onfirstaccess__' in self.__map__:
            target = self.__map__.pop('__onfirstaccess__')
            importobj(*target)()
        try:
            modpath, attrname = self.__map__[name]
        except KeyError:
            if target is not None and name != '__onfirstaccess__':
                # retry, onfirstaccess might have set attrs
                return getattr(self, name)
            raise AttributeError(name)
        else:
            result = importobj(modpath, attrname)
            setattr(self, name, result)
            try:
                del self.__map__[name]
            except KeyError:
                pass  # in a recursive-import situation a double-del can happen
            return result