def clone_to(self, target_catalog):
        """need to clone both the enclosed object + the wrapping asset"""
        def _clone(obj_id):
            package_name = obj_id.get_identifier_namespace().split('.')[0]
            obj_name = obj_id.get_identifier_namespace().split('.')[1].lower()

            catalogs = {
                'assessment': 'bank',
                'repository': 'repository'
            }

            my_catalog_name = catalogs[package_name.lower()]

            mgr = self.my_osid_object._get_provider_manager(package_name.upper())

            if self.my_osid_object._proxy is not None:
                catalog_lookup_session = getattr(mgr, 'get_' + my_catalog_name + '_lookup_session')(
                    proxy=self.my_osid_object._proxy
                )
            else:
                catalog_lookup_session = getattr(mgr, 'get_' + my_catalog_name + '_lookup_session')()

            catalog = getattr(catalog_lookup_session, 'get_' + my_catalog_name)(target_catalog.ident)

            if self.my_osid_object._proxy is not None:
                admin_session = getattr(mgr, 'get_' + obj_name + '_admin_session_for_' + my_catalog_name)(
                    catalog.ident,
                    proxy=self.my_osid_object._proxy
                )
            else:
                admin_session = getattr(mgr, 'get_' + obj_name + '_admin_session_for_' + my_catalog_name)(catalog.ident)

            new_obj = getattr(admin_session, 'duplicate_' + obj_name)(obj_id)
            try:
                form = getattr(admin_session, 'get_' + obj_name + '_form_for_update')(new_obj.ident)
                form.set_provenance(str(obj_id))
                new_obj = getattr(admin_session, 'update_' + obj_name)(form)
            except AttributeError:
                pass
            return new_obj

        enclosed_object_id = self.get_enclosed_object_id()
        new_enclosed_object = _clone(enclosed_object_id)

        # Let's do the asset here
        enclosure_id = self.my_osid_object.ident
        new_asset = _clone(enclosure_id)
        package_name = enclosure_id.get_identifier_namespace().split('.')[0]
        obj_name = enclosure_id.get_identifier_namespace().split('.')[1].lower()

        catalogs = {
            'assessment': 'bank',
            'repository': 'repository'
        }

        my_catalog_name = catalogs[package_name.lower()]

        mgr = self.my_osid_object._get_provider_manager(package_name.upper())

        if self.my_osid_object._proxy is not None:
            catalog_lookup_session = getattr(mgr, 'get_' + my_catalog_name + '_lookup_session')(
                proxy=self.my_osid_object._proxy
            )
        else:
            catalog_lookup_session = getattr(mgr, 'get_' + my_catalog_name + '_lookup_session')()
        catalog = getattr(catalog_lookup_session, 'get_' + my_catalog_name)(target_catalog.ident)

        if self.my_osid_object._proxy is not None:
            admin_session = getattr(mgr, 'get_' + obj_name + '_admin_session_for_' + my_catalog_name)(
                catalog.ident,
                proxy=self.my_osid_object._proxy
            )
        else:
            admin_session = getattr(mgr, 'get_' + obj_name + '_admin_session_for_' + my_catalog_name)(catalog.ident)

        form = getattr(admin_session, 'get_' + obj_name + '_form_for_update')(new_asset.ident)
        form.set_enclosed_object(new_enclosed_object.ident)
        return getattr(admin_session, 'update_' + obj_name)(form)