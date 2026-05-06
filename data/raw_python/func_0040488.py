def _parsecsv(x):
        """Deserialize file-like object containing csv to a Python generator.
        """
        for line in x:
            # decode as utf-8, whitespace-strip and split on delimiter
            yield line.decode('utf-8').strip().split(config.DELIMITER)