def check_required_fields(self, ignore_fields=list(), allow_no_resources=False):
        # type: (List[str], bool) -> None
        """Check that metadata for dataset and its resources is complete. The parameter ignore_fields
        should be set if required to any fields that should be ignored for the particular operation.

        Args:
            ignore_fields (List[str]): Fields to ignore. Default is [].
            allow_no_resources (bool): Whether to allow no resources. Defaults to False.

        Returns:
            None
        """
        if self.is_requestable():
            self._check_required_fields('dataset-requestable', ignore_fields)
        else:
            self._check_required_fields('dataset', ignore_fields)
            if len(self.resources) == 0 and not allow_no_resources:
                raise HDXError('There are no resources! Please add at least one resource!')
            for resource in self.resources:
                ignore_fields = ['package_id']
                resource.check_required_fields(ignore_fields=ignore_fields)