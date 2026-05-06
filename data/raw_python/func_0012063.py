def dump(self, path: str, file_name: str = "", **kwargs: dict):
        """
        Dumps the entire index into a json file. 

        :param path: The path to directory where the dump should be stored.
        :param file_name: Name of the file the dump should be stored in. If empty the index name is used.
        :param kwargs: Keyword arguments for the json converter. (ex. indent=4, ensure_ascii=False)
        """
        export = list()
        for results in self.scroll():
            export.extend(results)

        if not path.endswith('/'):
            path += '/'

        if file_name == '':
            file_name = self.index

        if not file_name.endswith('.json'):
            file_name += '.json'

        store = path + file_name
        with open(store, 'w') as fp:
            json.dump(export, fp, **kwargs)

        logging.info("Extracted %s records from the index %s and stored them in %s/%s.", len(export), self.index, path, file_name)