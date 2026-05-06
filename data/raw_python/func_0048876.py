def get_enclosed_object(self):
        """Return the enclosed object"""
        if self._enclosed_object is None:
            enclosed_object_id = self.get_enclosed_object_id()
            package_name = enclosed_object_id.get_identifier_namespace().split('.')[0]
            obj_name = enclosed_object_id.get_identifier_namespace().split('.')[1]
            mgr = self.my_osid_object._get_provider_manager(package_name.upper())
            try:
                lookup_session = getattr(mgr, 'get_' + obj_name.lower() + '_lookup_session')(self.my_osid_object._proxy)
            except TypeError:
                lookup_session = getattr(mgr, 'get_' + obj_name.lower() + '_lookup_session')()
            getattr(lookup_session, 'use_federated_' + CATALOG_LOOKUP[package_name] + '_view')()
            self._enclosed_object = getattr(
                lookup_session, 'get_' + obj_name.lower())(enclosed_object_id)
        return self._enclosed_object