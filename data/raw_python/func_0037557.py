def hash_file(f):
        '''
        Utility function for hashing file contents

        Overload this function to change the file equality checking algorithm

        Parameters
        ----------

        f: file-like
            File-like object or file path from which to compute checksum value


        Returns
        -------
        checksum: dict
            dictionary with {'algorithm': 'md5', 'checksum': hexdigest}

        '''

        md5 = hashlib.md5()

        with open_filelike(f, 'rb') as f_obj:
            for chunk in iter(lambda: f_obj.read(128 * md5.block_size), b''):
                md5.update(chunk)

        return {'algorithm': 'md5', 'checksum': md5.hexdigest()}