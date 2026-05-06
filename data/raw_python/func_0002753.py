def _execute_connector(connector_command, top_level_argument, *file_contents, listing=None):
        """
        Executes a connector by executing the given connector_command. The content of args will be the content of the
        files handed to the connector cli.

        :param connector_command: The connector command to execute.
        :param top_level_argument: The top level command line argument for the connector cli.
        (Like 'receive' or 'send_validate')
        :param file_contents: A dict of information handed over to the connector cli.
        :param listing: A listing to provide to the connector cli. Will be ignored if None.
        :return: A tuple containing the return code of the connector and the stderr of the command as str.
        """
        # create temp_files for every file_content
        temp_files = []
        for file_content in file_contents:
            if file_content is None:
                continue
            tmp_file = tempfile.NamedTemporaryFile('w')
            json.dump(file_content, tmp_file)
            tmp_file.flush()
            temp_files.append(tmp_file)

        tmp_listing_file = None
        if listing:
            tmp_listing_file = tempfile.NamedTemporaryFile('w')
            json.dump(listing, tmp_listing_file)
            tmp_listing_file.flush()

        command = [connector_command, top_level_argument]
        command.extend([t.name for t in temp_files])

        if tmp_listing_file:
            command.append('--listing {}'.format(tmp_listing_file.name))

        result = execute(' '.join(command))

        # close temp_files
        for temp_file in temp_files:
            temp_file.close()

        if tmp_listing_file:
            tmp_listing_file.close()

        return result['returnCode'], result['stdErr']