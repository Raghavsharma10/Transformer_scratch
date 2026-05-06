def download_member_shared(cls, member_data, target_member_dir, source=None,
                               max_size=MAX_SIZE_DEFAULT, id_filename=False):
        """
        Download files to sync a local dir to match OH member shared data.

        Files are downloaded to match their "basename" on Open Humans.
        If there are multiple files with the same name, the most recent is
        downloaded.

        :param member_data: This field is data related to member in a project.
        :param target_member_dir: This field is the target directory where data
            will be downloaded.
        :param source: This field is the source from which to download data.
        :param max_size: This field is the maximum file size. It's default
            value is 128m.
        """
        logging.debug('Download member shared data...')
        sources_shared = member_data['sources_shared']
        file_data = cls._get_member_file_data(member_data,
                                              id_filename=id_filename)

        logging.info('Downloading member data to {}'.format(target_member_dir))
        for basename in file_data:

            # If not in sources shared, it's the project's own data. Skip.
            if file_data[basename]['source'] not in sources_shared:
                continue

            # Filter source if specified. Determine target directory for file.
            if source:
                if source == file_data[basename]['source']:
                    target_filepath = os.path.join(target_member_dir, basename)
                else:
                    continue
            else:
                source_data_dir = os.path.join(target_member_dir,
                                               file_data[basename]['source'])
                if not os.path.exists(source_data_dir):
                    os.mkdir(source_data_dir)
                target_filepath = os.path.join(source_data_dir, basename)

            download_file(download_url=file_data[basename]['download_url'],
                          target_filepath=target_filepath,
                          max_bytes=parse_size(max_size))