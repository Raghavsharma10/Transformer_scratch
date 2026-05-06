def pack(self, fname, source_dir=None, password=None, pickle_fname=None):
        """
        High level function for repacking a backup file from the given
        target directory (will be generated based on the filename if not given).

        Requires also a filename.pickle file which was generated during the unpacking
        step.

        The fields `version`, `compression` and `encryption` have to be set before calling
        this method.

        :param source_dir: the directory to create the backup file from
                           (default: filename + _unpacked)
        :param password: optional password for decrypting the backup
                         (can also be set in the constructor)
        """
        if source_dir is None:
            source_dir = os.path.basename(fname) + '_unpacked'
        if pickle_fname is None:
            pickle_fname = os.path.basename(fname) + '.pickle'

        assert self.version is not None, "Backup version is not set"
        assert self.compression is not None, "Compression level is not set"
        assert self.encryption is not None, "Encryption level is not set"

        data = io.BytesIO()
        tar = tarfile.TarFile(name=fname,
                              fileobj=data,
                              mode='w',
                              format=tarfile.PAX_FORMAT)

        with open(pickle_fname, 'rb') as fp:
            members = pickle.load(fp)

        with open(fname, 'wb') as fp:
            os.chdir(source_dir)
            for member in members:
                if member.isreg():
                    tar.addfile(member, open(member.name, 'rb'))
                else:
                    tar.addfile(member)

            tar.close()

            data.seek(0)
            if self.compression == CompressionType.ZLIB:
                compressor = zlib.compressobj(method=zlib.DEFLATED)
                data = compressor.compress(data.read()) + compressor.flush()
            if self.is_encrypted():
                data = self._encrypt(data, password=password)
        
            fp.write(b'ANDROID BACKUP\n')
            fp.write('{}\n'.format(self.version).encode())
            fp.write('{:d}\n'.format(self.compression).encode())
            fp.write('{}\n'.format(self.encryption.value).encode())

            fp.write(data)