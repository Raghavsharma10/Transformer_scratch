def new(self, page_name, **dict):
        '''
        Create a new item with the provided dict information
        at the given page_name.  Returns the new item.

        As of version 2.2 of Redmine, this doesn't seem to function.
        '''
        self._item_new_path = '/projects/%s/wiki/%s.json' % \
            (self._project.identifier, page_name)
        # Call the base class new method
        return super(Redmine_Wiki_Pages_Manager, self).new(**dict)