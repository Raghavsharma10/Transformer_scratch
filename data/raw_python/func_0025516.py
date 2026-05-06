def read_properties(self):
        """
            Read properties from the ndata file reference

            :param reference: the reference from which to read
            :return: a tuple of the item_uuid and a dict of the properties
        """
        with self.__lock:
            absolute_file_path = self.__file_path
            with open(absolute_file_path, "rb") as fp:
                local_files, dir_files, eocd = parse_zip(fp)
                properties = read_json(fp, local_files, dir_files, b"metadata.json")
            return properties