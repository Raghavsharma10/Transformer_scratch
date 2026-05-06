def get_remote_file(self, remote_path, local_path):
        """Fetch remote File.

        :param remote_path: remote path
        :param local_path: local path
        """
        sftp_client = self.transport.open_sftp_client()
        LOG.debug('Get the remote file. '
                  'Source=%(src)s. Target=%(target)s.' %
                  {'src': remote_path, 'target': local_path})
        try:
            sftp_client.get(remote_path, local_path)
        except Exception as ex:
            LOG.error('Failed to secure copy. Reason: %s.' %
                      six.text_type(ex))
            raise SFtpExecutionError(err=ex)