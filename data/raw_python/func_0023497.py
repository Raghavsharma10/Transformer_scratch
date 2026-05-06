def download_file(self, remote_path, local_path, progress=None):
        """Downloads file from WebDAV server and save it locally.
        More information you can find by link http://webdav.org/specs/rfc4918.html#rfc.section.9.4

        :param remote_path: the path to remote file for downloading.
        :param local_path: the path to save file locally.
        :param progress: progress function. Not supported now.
        """
        urn = Urn(remote_path)
        if self.is_dir(urn.path()):
            raise OptionNotValid(name='remote_path', value=remote_path)

        if os.path.isdir(local_path):
            raise OptionNotValid(name='local_path', value=local_path)

        if os.path.sep in local_path:
            os.makedirs(local_path.rsplit(os.path.sep, 1)[0], exist_ok=True)

        if not self.check(urn.path()):
            raise RemoteResourceNotFound(urn.path())

        with open(local_path, 'wb') as local_file:
            response = self.execute_request('download', urn.quote())
            for block in response.iter_content(1024):
                local_file.write(block)