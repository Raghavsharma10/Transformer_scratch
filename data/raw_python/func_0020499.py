def import_image_tags(self, name, stream_import, tags, repository, insecure):
        """
        Import image tags from a Docker registry into an ImageStream

        :return: bool, whether tags were imported
        """

        # Get the JSON for the ImageStream
        imagestream_json = self.get_image_stream(name).json()
        logger.debug("imagestream: %r", imagestream_json)
        changed = False

        # existence of dockerImageRepository is limiting how many tags are updated
        if 'dockerImageRepository' in imagestream_json.get('spec', {}):
            logger.debug("Removing 'dockerImageRepository' from ImageStream %s", name)
            imagestream_json['spec'].pop('dockerImageRepository')
            changed = True
        all_annotations = imagestream_json.get('metadata', {}).get('annotations', {})
        # remove annotations about registry, since method will get them as arguments
        for annotation in ANNOTATION_SOURCE_REPO, ANNOTATION_INSECURE_REPO:
            if annotation in all_annotations:
                imagestream_json['metadata']['annotations'].pop(annotation)
                changed = True

        if changed:
            imagestream_json = self.update_image_stream(name, imagestream_json).json()

        # Note the tags before import
        oldtags = imagestream_json.get('status', {}).get('tags', [])
        logger.debug("tags before import: %r", oldtags)

        stream_import['metadata']['name'] = name
        stream_import['spec']['images'] = []
        tags_set = set(tags) if tags else set()

        if not tags_set:
            logger.debug('No tags to import')
            return False

        for tag in tags_set:
            image_import = {
                'from': {"kind": "DockerImage",
                         "name": '{}:{}'.format(repository, tag)},
                'to': {'name': tag},
                'importPolicy': {'insecure': insecure},
                # referencePolicy will default to "type: source"
                # so we don't have to explicitly set it
            }
            stream_import['spec']['images'].append(image_import)

        import_url = self._build_url("imagestreamimports/")
        import_response = self._post(import_url, data=json.dumps(stream_import),
                                     use_json=True)
        self._check_import_image_response(import_response)

        new_tags = [
            image['tag']
            for image in import_response.json().get('status', {}).get('images', [])]
        logger.debug("tags after import: %r", new_tags)

        return True