def get_interface_by_instance_name(self, name):
        '''Get an interface of this port by instance name.'''
        with self._mutex:
            for intf in self.interfaces:
                if intf.instance_name == name:
                    return intf
            return None