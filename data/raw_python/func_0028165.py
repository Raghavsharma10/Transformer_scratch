def _update_resource_view(self, log=False):
        # type: () -> bool
        """Check if resource view exists in HDX and if so, update resource view

        Returns:
            bool: True if updated and False if not
        """
        update = False
        if 'id' in self.data and self._load_from_hdx('resource view', self.data['id']):
            update = True
        else:
            if 'resource_id' in self.data:
                resource_views = self.get_all_for_resource(self.data['resource_id'])
                for resource_view in resource_views:
                    if self.data['title'] == resource_view['title']:
                        self.old_data = self.data
                        self.data = resource_view.data
                        update = True
                        break
        if update:
            if log:
                logger.warning('resource view exists. Updating %s' % self.data['id'])
            self._merge_hdx_update('resource view', 'id')
        return update