def interfaces(self):
        '''The list of interfaces this port provides or uses.

        This list will be created at the first reference to this property.
        This means that the first reference may be delayed by CORBA calls,
        but others will return quickly (unless a delayed reparse has been
        triggered).

        '''
        with self._mutex:
            if not self._interfaces:
                profile = self._obj.get_port_profile()
                self._interfaces = [SvcInterface(intf) \
                                    for intf in profile.interfaces]
        return self._interfaces