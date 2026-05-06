def ensure_image_stream_tag(self, stream, tag_name, scheduled=False,
                                source_registry=None, organization=None, base_image=None):
        """Ensures the tag is monitored in ImageStream

        :param stream: dict, ImageStream object
        :param tag_name: str, name of tag to check, without name of
                              ImageStream as prefix
        :param scheduled: bool, if True, importPolicy.scheduled will be
                                set to True in ImageStreamTag
        :param source_registry: dict, info about source registry
        :param organization: str, oganization for registry
        :param base_image: str, base image
        :return: bool, whether or not modifications were performed
        """
        img_stream_tag_file = os.path.join(self.os_conf.get_build_json_store(),
                                           'image_stream_tag.json')
        with open(img_stream_tag_file) as f:
            tag_template = json.load(f)

        repository = None
        registry = None
        insecure = False

        if source_registry:
            registry = RegistryURI(source_registry['url']).docker_uri
            insecure = source_registry.get('insecure', False)

        if base_image and registry:
            repository = self._get_enclosed_repo_with_source_registry(base_image,
                                                                      registry, organization)

        return self.os.ensure_image_stream_tag(stream, tag_name, tag_template,
                                               scheduled, repository=repository,
                                               insecure=insecure)