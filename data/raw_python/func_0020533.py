def import_image(self, name, tags=None):
        """
        Import image tags from a Docker registry into an ImageStream

        :return: bool, whether tags were imported
        """
        stream_import_file = os.path.join(self.os_conf.get_build_json_store(),
                                          'image_stream_import.json')
        with open(stream_import_file) as f:
            stream_import = json.load(f)
        return self.os.import_image(name, stream_import, tags=tags)