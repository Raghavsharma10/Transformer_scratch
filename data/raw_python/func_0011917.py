def download_document(self, document: Document, overwrite=True, path=None):
        """
        Download a document to the given path. if no path is provided the path is constructed frome the base_url + stud.ip path + filename.
        If overwrite is set the local version will be overwritten if the file was changed on studip since the last check
        """
        if not path:
            path = os.path.join(os.path.expanduser(c["base_path"]), document.path)
        if (self.modified(document) and overwrite) or not os.path.exists(join(path, document.title)):
            log.info("Downloading %s" % join(path, document.title))
            file = self._get('/api/documents/%s/download' % document.id, stream=True)
            os.makedirs(path, exist_ok=True)
            with open(join(path, document.title), 'wb') as f:
                shutil.copyfileobj(file.raw, f)