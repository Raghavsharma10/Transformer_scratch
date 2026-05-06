def _check_required_fields(self, object_type, ignore_fields):
        # type: (str, List[str]) -> None
        """Helper method to check that metadata for HDX object is complete

        Args:
            ignore_fields (List[str]): Any fields to ignore in the check

        Returns:
            None
        """
        for field in self.configuration[object_type]['required_fields']:
            if field not in self.data and field not in ignore_fields:
                raise HDXError('Field %s is missing in %s!' % (field, object_type))