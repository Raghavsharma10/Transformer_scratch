def parent(self):
        "Get this object's parent"
        if self._parent:
            return self._parent
        # auto-compute parent if needed
        elif getattr(self, '__parent_type__', None):
            return self._get_subfolder('..' if self._url[2].endswith('/')
                                            else '.', self.__parent_type__)
        else:
            raise AttributeError("%r has no parent attribute" % type(self))