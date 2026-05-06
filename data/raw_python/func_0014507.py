def register(self, obj, resource):
        """
        Register a resource to be closed with an object is collected.
        
        When the given C{obj} is garbage-collected by the python interpreter,
        the C{resource} will be closed by having its C{close()} method called.
        Any exceptions are ignored.
        
        @param obj: the object to track
        @type obj: object
        @param resource: the resource to close when the object is collected
        @type resource: object
        """
        def callback(ref):
            try:
                resource.close()
            except:
                pass
            del self._table[id(resource)]

        # keep the weakref in a table so it sticks around long enough to get
        # its callback called. :)
        self._table[id(resource)] = weakref.ref(obj, callback)