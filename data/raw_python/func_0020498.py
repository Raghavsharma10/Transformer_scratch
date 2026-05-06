def import_image(self, name, stream_import, tags=None):
        """
        Import image tags from a Docker registry into an ImageStream

        :return: bool, whether tags were imported
        """

        # Get the JSON for the ImageStream
        imagestream_json = self.get_image_stream(name).json()
        logger.debug("imagestream: %r", imagestream_json)

        if 'dockerImageRepository' in imagestream_json.get('spec', {}):
            logger.debug("Removing 'dockerImageRepository' from ImageStream %s", name)
            source_repo = imagestream_json['spec'].pop('dockerImageRepository')
            imagestream_json['metadata']['annotations'][ANNOTATION_SOURCE_REPO] = source_repo
            imagestream_json = self.update_image_stream(name, imagestream_json).json()

        # Note the tags before import
        oldtags = imagestream_json.get('status', {}).get('tags', [])
        logger.debug("tags before import: %r", oldtags)

        stream_import['metadata']['name'] = name
        stream_import['spec']['images'] = []
        tags_set = set(tags) if tags else set()
        for tag in imagestream_json.get('spec', {}).get('tags', []):
            if tags_set and tag['name'] not in tags_set:
                continue

            image_import = {
                'from': tag['from'],
                'to': {'name': tag['name']},
                'importPolicy': tag.get('importPolicy'),
                'referencePolicy': tag.get('referencePolicy'),
            }
            stream_import['spec']['images'].append(image_import)

        if not stream_import['spec']['images']:
            logger.debug('No tags to import')
            return False

        import_url = self._build_url("imagestreamimports/")
        import_response = self._post(import_url, data=json.dumps(stream_import),
                                     use_json=True)
        self._check_import_image_response(import_response)

        new_tags = [
            image['tag']
            for image in import_response.json().get('status', {}).get('images', [])]
        logger.debug("tags after import: %r", new_tags)

        return True