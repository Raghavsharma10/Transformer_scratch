def stash(self, storage, url):
        """Stores the uploaded file in a temporary storage location."""
        result = {}
        if self.is_valid():
            upload = self.cleaned_data['upload']
            name = storage.save(upload.name, upload)
            result['filename'] = os.path.basename(name)
            try:
                result['url'] = storage.url(name)
            except NotImplementedError:
                result['url'] = None
            result['stored'] = serialize_upload(name, storage, url)
        return result