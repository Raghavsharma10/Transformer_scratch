def load_from_JSON(json_filename):
        '''
        Create a new instance of a PIDClientCredentials with information read
        from a local JSON file.

        :param json_filename: The path to the json credentials file. The json
            file should have the following format:

                .. code:: json

                    {
                        "handle_server_url": "https://url.to.your.handle.server",
                        "username": "index:prefix/suffix",
                        "password": "ZZZZZZZ",
                        "prefix": "prefix_to_use_for_writing_handles",
                        "handleowner": "username_to_own_handles"
                    }

            Any additional key-value-pairs are stored in the instance as
            config.
        :raises: :exc:`~b2handle.handleexceptions.CredentialsFormatError`
        :raises: :exc:`~b2handle.handleexceptions.HandleSyntaxError`
        :return: An instance.
        '''
        try:
            jsonfilecontent = json.loads(open(json_filename, 'r').read())
        except ValueError as exc:
            raise CredentialsFormatError(msg="Invalid JSON syntax: "+str(exc))
        instance = PIDClientCredentials(credentials_filename=json_filename,**jsonfilecontent)
        return instance