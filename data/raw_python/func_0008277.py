def get_object(conn, binding_name, object_cls):
        """
        Get a reference to a remote object using CORBA
        """
        try:
            obj = conn.rootContext.resolve(binding_name)
            narrowed = obj._narrow(object_cls)
        except CORBA.TRANSIENT:
            raise IOError('Attempt to retrieve object failed')

        if narrowed is None:
            raise IOError('Attempt to retrieve object got a different class of object')

        return narrowed