def get_object(self, binding_name, cls):
        """
        Get a reference to a remote object using CORBA
        """
        return self._state.get_object(self, binding_name, cls)