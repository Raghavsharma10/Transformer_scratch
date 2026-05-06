def _augment_file(self, f):
        """
        Augment a FileRecord with methods to get the data URL and to download, returning the updated file for use
        in generator functions
        :internal:
        """

        def get_url(target):
            if target.file_size is None:
                return None
            if target.file_name is not None:
                return self.base_url + '/files/content/{0}/{1}'.format(target.file_id.hex, target.file_name)
            else:
                return self.base_url + '/files/content/{0}'.format(target.file_id.hex, )

        f.get_url = types.MethodType(get_url, f)

        def download_to(target, file_name):
            url = target.get_url()
            r = requests.get(url, stream=True)
            with open(file_name, 'wb') as file_to_write:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        file_to_write.write(chunk)
                        file_to_write.flush()
            return file_name

        f.download_to = types.MethodType(download_to, f)
        return f