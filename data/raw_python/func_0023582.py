def upload_check(self, filename=None, folder_key=None, filedrop_key=None,
                     size=None, hash_=None, path=None, resumable=None):
        """upload/check

        http://www.mediafire.com/developers/core_api/1.3/upload/#check
        """
        return self.request('upload/check', QueryParams({
            'filename': filename,
            'folder_key': folder_key,
            'filedrop_key': filedrop_key,
            'size': size,
            'hash': hash_,
            'path': path,
            'resumable': resumable
        }))