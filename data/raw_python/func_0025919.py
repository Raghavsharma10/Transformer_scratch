def _add_item_manager(self, key, item_class, **paths):
        '''
        Add an item manager to this object.
        '''
        updated_paths = {}
        for path_type, path_value in paths.iteritems():
            updated_paths[path_type] = path_value.format(**self.__dict__)

        manager = Redmine_Items_Manager(self._redmine, item_class,
                                        **updated_paths)
        self.__dict__[key] = manager