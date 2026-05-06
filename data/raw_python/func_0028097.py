def reorder_resources(self, resource_ids, hxl_update=True):
        # type: (List[str], bool) -> None
        """Reorder resources in dataset according to provided list.
        If only some resource ids are supplied then these are
        assumed to be first and the other resources will stay in
        their original order.

        Args:
            resource_ids (List[str]): List of resource ids
            hxl_update (bool): Whether to call package_hxl_update. Defaults to True.

        Returns:
            None
        """
        dataset_id = self.data.get('id')
        if not dataset_id:
            raise HDXError('Dataset has no id! It must be read, created or updated first.')
        data = {'id': dataset_id,
                'order': resource_ids}
        self._write_to_hdx('reorder', data, 'package_id')
        if hxl_update:
            self.hxl_update()