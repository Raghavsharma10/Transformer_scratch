def import_image_tags(self, name, tags, repository, insecure=False):
        """Import image tags from specified container repository.

        :param name: str, name of ImageStream object
        :param tags: iterable, tags to be imported
        :param repository: str, remote location of container image
                                in the format <registry>/<repository>
        :param insecure: bool, indicates whenever registry is secure

        :return: bool, whether tags were imported
        """
        stream_import_file = os.path.join(self.os_conf.get_build_json_store(),
                                          'image_stream_import.json')
        with open(stream_import_file) as f:
            stream_import = json.load(f)
        return self.os.import_image_tags(name, stream_import, tags,
                                         repository, insecure)