def create_image_stream(self, name, docker_image_repository,
                            insecure_registry=False):
        """
        Create an ImageStream object

        Raises exception on error

        :param name: str, name of ImageStream
        :param docker_image_repository: str, pull spec for docker image
               repository
        :param insecure_registry: bool, whether plain HTTP should be used
        :return: response
        """
        img_stream_file = os.path.join(self.os_conf.get_build_json_store(), 'image_stream.json')
        with open(img_stream_file) as f:
            stream = json.load(f)
        stream['metadata']['name'] = name
        stream['metadata'].setdefault('annotations', {})
        stream['metadata']['annotations'][ANNOTATION_SOURCE_REPO] = docker_image_repository
        if insecure_registry:
            stream['metadata']['annotations'][ANNOTATION_INSECURE_REPO] = 'true'

        return self.os.create_image_stream(json.dumps(stream))