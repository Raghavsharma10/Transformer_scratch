def _replace_data_types(dictionary):
        """
        Replaces strings with appropriate data types (int, boolean).
        Also replaces the human readable logging levels with the integer form.

        dictionary: A dictionary returned from the config file.
        """
        logging_levels = {'NONE': 0, 'NULL': 0, 'DEBUG': 10, 'INFO': 20,
                'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}
        for k, v in dictionary.items():
            if v in ['true', 'True', 'on']:
                dictionary[k] = True
            elif v in ['false', 'False', 'off']:
                dictionary[k] = False
            elif k == 'log_file' and '~' in v:
                dictionary[k] = v.replace('~', os.path.expanduser('~'))
            elif v in logging_levels:
                dictionary[k] = logging_levels[v]
            elif isinstance(v, str) and v.isnumeric():
                dictionary[k] = int(v)
            elif ',' in v:
                dictionary[k] = [x.strip() for x in v.split(',')]
        return dictionary