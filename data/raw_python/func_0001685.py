def _load_json_file(json_file):
        """ load json file and check file content format
        """
        with io.open(json_file, encoding='utf-8') as data_file:
            try:
                json_content = json.load(data_file)
            except p_exception.JSONDecodeError:
                err_msg = u"JSONDecodeError: JSON file format error: {}".format(json_file)
                raise p_exception.FileFormatError(err_msg)

            FileUtils._check_format(json_file, json_content)
            return json_content