def download_member_project_data(cls, member_data, target_member_dir,
                                     max_size=MAX_SIZE_DEFAULT,
                                     id_filename=False):
        """
        Download files to sync a local dir to match OH member project data.

        :param member_data: This field is data related to member in a project.
        :param target_member_dir: This field is the target directory where data
            will be downloaded.
        :param max_size: This field is the maximum file size. It's default
            value is 128m.
        """
        logging.debug('Download member project data...')
        sources_shared = member_data['sources_shared']
        file_data = cls._get_member_file_data(member_data,
                                              id_filename=id_filename)
        for basename in file_data:
            # This is using a trick to identify a project's own data in an API
            # response, without knowing the project's identifier: if the data
            # isn't a shared data source, it must be the project's own data.
            if file_data[basename]['source'] in sources_shared:
                continue
            target_filepath = os.path.join(target_member_dir, basename)
            download_file(download_url=file_data[basename]['download_url'],
                          target_filepath=target_filepath,
                          max_bytes=parse_size(max_size))