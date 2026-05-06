def readCfgJson(cls, working_path):
        """Read cmWalk configuration data of a working directory from a json file.

        :param working_path: working path for reading the configuration data.
        :return: the configuration data represented in a json object, None if the configuration files does not
                 exist.
        """

        cfg_json_filename = os.path.join(working_path, cls.CFG_JSON_FILENAME)
        if os.path.isfile(cfg_json_filename):
            with open(cfg_json_filename) as json_file:
                cfg = json.load(json_file)
                return cfg
        return None