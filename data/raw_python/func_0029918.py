def to_meta(self, md5=None, file=None):
        """Return a dictionary of metadata, for use in the Remote api."""
        # from collections import OrderedDict

        if not md5:
            if not file:
                raise ValueError('Must specify either file or md5')

            md5 = md5_for_file(file)
            size = os.stat(file).st_size
        else:
            size = None

        return {
            'id': self.id_,
            'identity': json.dumps(self.dict),
            'name': self.sname,
            'fqname': self.fqname,
            'md5': md5,
            # This causes errors with calculating the AWS signature
            'size': size
        }