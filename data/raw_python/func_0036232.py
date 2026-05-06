def get_file_specs(self, filepath, keep_folders=False):
        """Gather information on files needed for valid transfer.

        :param filepath: Path to file in question
        :param keep_folders: Whether or not to maintain folder structure
        :type keep_folders: bool
        :type filepath: str, unicode
        :rtype: ``dict``
        """

        path, filename = os.path.split(filepath)

        fileid = str(uuid4()).replace('-', '')

        if self.checksum:
            with open(filepath, 'rb') as f:
                md5hash = md5(f.read()).digest().encode('base64')[:-1]
        else:
            md5hash = None

        specs = {
            'transferid': self.transfer_id,
            'transferkey': self.transfer_info['transferkey'],
            'fileid': fileid,
            'filepath': filepath,
            'thefilename': keep_folders and filepath or filename,
            'totalsize': os.path.getsize(filepath),
            'md5': md5hash,
            'content-type': guess_type(filepath)[0]
            }

        return specs