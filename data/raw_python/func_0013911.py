def copy_file_to_remote(self, local_path, remote_path):
        """scp the local file to remote folder.

        :param local_path: local path
        :param remote_path: remote path
        """
        sftp_client = self.transport.open_sftp_client()
        LOG.debug('Copy the local file to remote. '
                  'Source=%(src)s. Target=%(target)s.' %
                  {'src': local_path, 'target': remote_path})
        try:
            sftp_client.put(local_path, remote_path)
        except Exception as ex:
            LOG.error('Failed to copy the local file to remote. '
                      'Reason: %s.' % six.text_type(ex))
            raise SFtpExecutionError(err=ex)