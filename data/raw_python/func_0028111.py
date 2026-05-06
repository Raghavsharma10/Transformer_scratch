def get_all_dataset_names(configuration=None, **kwargs):
        # type: (Optional[Configuration], Any) -> List[str]
        """Get all dataset names in HDX

        Args:
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.
            **kwargs: See below
            limit (int): Number of rows to return. Defaults to all dataset names.
            offset (int): Offset in the complete result for where the set of returned dataset names should begin

        Returns:
            List[str]: list of all dataset names in HDX
        """
        dataset = Dataset(configuration=configuration)
        dataset['id'] = 'all dataset names'  # only for error message if produced
        return dataset._write_to_hdx('list', kwargs, 'id')