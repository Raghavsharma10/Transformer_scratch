def _set_version(self, version):
        '''
        Set up this object based on the capabilities of the
        known versions of Redmine
        '''
        # Store the version we are evaluating
        self.version = version or None
        # To evaluate the version capabilities,
        # assume the best-case if no version is provided.
        version_check = version or 9999.0

        if version_check < 1.0:
            raise RedmineError('This library will only work with '
                               'Redmine version 1.0 and higher.')

        ## SECURITY AUGMENTATION
        # All versions support the key in the request
        #  (http://server/stuff.json?key=blah)
        # But versions 1.1 and higher can put the key in a header field
        # for better security.
        # If no version was provided (0.0) then assume we should
        # set the key with the request.
        self.key_in_header = version >= 1.1
        # it puts the key in the header or
        # it gets the hose, but not for 1.0.

        self.impersonation_supported = version_check >= 2.2
        self.has_project_memberships = version_check >= 1.4
        self.has_project_versions = version_check >= 1.3
        self.has_wiki_pages = version_check >= 2.2

        ## ITEM MANAGERS
        # Step through all the item managers by version
        # and instatiate and item manager for that item.
        for manager_version in self._item_managers_by_version:
            if version_check >= manager_version:
                managers = self._item_managers_by_version[manager_version]
                for attribute_name, item in managers.iteritems():
                    setattr(self, attribute_name,
                            Redmine_Items_Manager(self, item))