def set_expected_update_frequency(self, update_frequency):
        # type: (str) -> None
        """Set expected update frequency

        Args:
            update_frequency (str): Update frequency

        Returns:
            None
        """
        try:
            int(update_frequency)
        except ValueError:
            update_frequency = Dataset.transform_update_frequency(update_frequency)
        if not update_frequency:
            raise HDXError('Invalid update frequency supplied!')
        self.data['data_update_frequency'] = update_frequency