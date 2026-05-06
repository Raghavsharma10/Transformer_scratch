def _loadMetaFromJson(self, path):
        """
        Reads the json meta into memory.
        :return: the meta.
        """
        try:
            with (path / 'metadata.json').open() as infile:
                return json.load(infile)
        except FileNotFoundError:
            logger.error('Metadata does not exist at ' + str(path))
            return None