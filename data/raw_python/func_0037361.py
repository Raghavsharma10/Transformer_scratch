def _queryless_all(cls):
        '''
        This is a hack because some datastore implementations don't support
        querying. Right now the solution is to drop down to the underlying
        native client and query all, which means that this section is ugly.
        If it were architected properly, you might be able to do something
        like inject an implementation of a NativeClient interface, which would
        let Switchboard users write their own NativeClient wrappers that
        implement all. However, at this point I'm just happy getting datastore
        to work, so quick-and-dirty will suffice.
        '''
        if hasattr(cls.ds, '_redis'):
            r = cls.ds._redis
            keys = r.keys()
            serializer = cls.ds.child_datastore.serializer

            def get_value(k):
                value = r.get(k)
                return value if value is None else serializer.loads(value)
            return [get_value(k) for k in keys]
        else:
            raise NotImplementedError