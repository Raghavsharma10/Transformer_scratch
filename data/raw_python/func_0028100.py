def read_from_hdx(identifier, configuration=None):
        # type: (str, Optional[Configuration]) -> Optional['Dataset']
        """Reads the dataset given by identifier from HDX and returns Dataset object

        Args:
            identifier (str): Identifier of dataset
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.

        Returns:
            Optional[Dataset]: Dataset object if successful read, None if not
        """

        dataset = Dataset(configuration=configuration)
        result = dataset._dataset_load_from_hdx(identifier)
        if result:
            return dataset
        return None