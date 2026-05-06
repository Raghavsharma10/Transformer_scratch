def export_file(self, data_object, destination_directory=None,
                    destination_filename=None, retry=False,
                    export_metadata=False, export_raw_file=True):
        """Export a file from Loom to some file storage location.
        Default destination_directory is cwd. Default destination_filename is the 
        filename from the file data object associated with the given file_id.
        """
        if not destination_directory:
            destination_directory = os.getcwd()

        # We get filename from the dataobject
        if not destination_filename:
            destination_filename = data_object['value']['filename']

        destination_file_url = os.path.join(destination_directory,
                                            destination_filename)

        logger.info('Exporting file %s@%s ...' % (
            data_object['value']['filename'],
            data_object['uuid']))

        if export_raw_file:
            destination = File(
                destination_file_url, self.storage_settings, retry=retry)
            if destination.exists():
                raise FileAlreadyExistsError(
                    'File already exists at %s' % destination_file_url)
            logger.info('...copying file to %s' % (
                destination.get_url()))

            # Copy from the first file location
            file_resource = data_object.get('value')
            md5 = file_resource.get('md5')
            source_url = data_object['value']['file_url']
            File(source_url, self.storage_settings, retry=retry).copy_to(
                destination, expected_md5=md5)
            data_object['value'] = self._create_new_file_resource(
                data_object['value'], destination.get_url())
        else:
            logger.info('...skipping raw file')

        if export_metadata:
            data_object['value'].pop('link', None)
            data_object['value'].pop('upload_status', None)
            destination_metadata_url = os.path.join(
                destination_file_url + '.metadata.yaml')
            logger.info('...writing metadata to %s' % destination_metadata_url)
            metadata = yaml.safe_dump(data_object, default_flow_style=False)
            metadata_file = File(destination_metadata_url,
                                 self.storage_settings, retry=retry)
            metadata_file.write(metadata)
        else:
            logger.info('...skipping metadata')

        logger.info('...finished file export')