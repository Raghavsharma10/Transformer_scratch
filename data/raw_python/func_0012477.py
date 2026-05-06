def _get_project_id(self):
        """
        Get our projectId from the ``GOOGLE_APPLICATION_CREDENTIALS`` creds
        JSON file.

        :return: project ID
        :rtype: str
        """
        fpath = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', None)
        if fpath is None:
            raise Exception('ERROR: No project ID specified, and '
                            'GOOGLE_APPLICATION_CREDENTIALS env var is not set')
        fpath = os.path.abspath(os.path.expanduser(fpath))
        logger.debug('Reading credentials file at %s to get project_id', fpath)
        with open(fpath, 'r') as fh:
            cred_data = json.loads(fh.read())
        return cred_data['project_id']