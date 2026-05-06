def get_parsed_query(self):
        """ Returns string with last query parsed. Assuming called after search_datasets."""
        return '{} OR {}'.format(
            self.backend.dataset_index.get_parsed_query()[0],
            self.backend.partition_index.get_parsed_query()[0])