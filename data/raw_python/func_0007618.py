def upload_member_from_dir(member_data, target_member_dir, metadata,
                               access_token, mode='default',
                               max_size=MAX_SIZE_DEFAULT):
        """
        Upload files in target directory to an Open Humans member's account.

        The default behavior is to overwrite files with matching filenames on
        Open Humans, but not otherwise delete files.

        If the 'mode' parameter is 'safe': matching filenames will not be
        overwritten.

        If the 'mode' parameter is 'sync': files on Open Humans that are not
        in the local directory will be deleted.

        :param member_data: This field is data related to member in a project.
        :param target_member_dir: This field is the target directory from where
            data will be uploaded.
        :param metadata: This field is metadata for files to be uploaded.
        :param access_token: This field is user specific access token.
        :param mode: This field takes three value default, sync, safe. It's
            default value is 'default'.
        :param max_size: This field is the maximum file size. It's default
            value is 128m.
        """
        if not validate_metadata(target_member_dir, metadata):
            raise ValueError('Metadata should match directory contents!')
        project_data = {f['basename']: f for f in member_data['data'] if
                        f['source'] not in member_data['sources_shared']}
        for filename in metadata:
            if filename in project_data and mode == 'safe':
                logging.info('Skipping {}, remote exists with matching'
                             ' name'.format(filename))
                continue
            filepath = os.path.join(target_member_dir, filename)
            remote_file_info = (project_data[filename] if filename in
                                project_data else None)
            upload_aws(target_filepath=filepath,
                       metadata=metadata[filename],
                       access_token=access_token,
                       project_member_id=member_data['project_member_id'],
                       remote_file_info=remote_file_info)
        if mode == 'sync':
            for filename in project_data:
                if filename not in metadata:
                    logging.debug("Deleting {}".format(filename))
                    delete_file(
                        file_basename=filename,
                        access_token=access_token,
                        project_member_id=member_data['project_member_id'])