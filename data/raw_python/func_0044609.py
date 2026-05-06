def add_bundle(self, data: dict) -> models.Bundle:
        """Build a new bundle version of files.

        The format of the input dict is defined in the `schema` module.
        """
        bundle_obj = self.bundle(data['name'])
        if bundle_obj and self.version(bundle_obj.name, data['created']):
            LOG.debug('version of bundle already added')
            return None

        if bundle_obj is None:
            bundle_obj = self.new_bundle(name=data['name'], created_at=data['created'])

        version_obj = self.new_version(created_at=data['created'], expires_at=data.get('expires'))

        tag_names = set(tag_name for file_data in data['files'] for tag_name in file_data['tags'])
        tag_map = self._build_tags(tag_names)

        for file_data in data['files']:
            LOG.debug(f"adding file: {file_data['path']}")
            if not Path(file_data['path']).exists():
                raise FileNotFoundError(file_data['path'])
            tags = [tag_map[tag_name] for tag_name in file_data['tags']]
            new_file = self.new_file(file_data['path'], to_archive=file_data['archive'], tags=tags)
            version_obj.files.append(new_file)

        version_obj.bundle = bundle_obj
        return bundle_obj, version_obj