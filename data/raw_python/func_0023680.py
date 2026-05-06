def download_file(self, src_uri, target):
        """Download file from MediaFire.

        src_uri -- MediaFire file URI to download
        target -- download path or file-like object in write mode
        """
        resource = self.get_resource_by_uri(src_uri)
        if not isinstance(resource, File):
            raise MediaFireError("Only files can be downloaded")

        quick_key = resource['quickkey']
        result = self.api.file_get_links(quick_key=quick_key,
                                         link_type='direct_download')
        direct_download = result['links'][0]['direct_download']

        # Force download over HTTPS
        direct_download = direct_download.replace('http:', 'https:')

        name = resource['filename']

        target_is_filehandle = True if hasattr(target, 'write') else False

        if not target_is_filehandle:
            if (os.path.exists(target) and os.path.isdir(target)) or \
                    target.endswith("/"):
                target = os.path.join(target, name)

            if not os.path.isdir(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))

            logger.info("Downloading %s to %s", src_uri, target)

        response = requests.get(direct_download, stream=True)
        try:
            if target_is_filehandle:
                out_fd = target
            else:
                out_fd = open(target, 'wb')

            checksum = hashlib.sha256()
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    out_fd.write(chunk)
                    checksum.update(chunk)

            checksum_hex = checksum.hexdigest().lower()
            if checksum_hex != resource['hash']:
                raise DownloadError("Hash mismatch ({} != {})".format(
                    resource['hash'], checksum_hex))

            logger.info("Download completed successfully")
        finally:
            if not target_is_filehandle:
                out_fd.close()