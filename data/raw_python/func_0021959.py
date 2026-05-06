def list_folder(self, path):
        """Looks up folder contents of `path.`"""
        # Inspired by https://github.com/rspivak/sftpserver/blob/0.3/src/sftpserver/stub_sftp.py#L70
        try:
            folder_contents = []
            for f in os.listdir(path):
                attr = paramiko.SFTPAttributes.from_stat(os.stat(os.path.join(path, f)))
                attr.filename = f
                folder_contents.append(attr)
            return folder_contents
        except OSError as e:
            return SFTPServer.convert_errno(e.errno)