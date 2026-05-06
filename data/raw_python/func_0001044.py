def download(self, url, path, force_refetch=False, nocache=False):
        ''' Download a file at $url and save it to $path
        '''
        # Enable cache
        if os.path.isfile(path):
            getLogger().info("File exists, download task skipped -> {path}".format(path=path))
            return True
        try:
            # Open URL
            getLogger().info("Downloading: {url} -> {path}".format(url=url, path=path))
            response = self.fetch(url, force_refetch=force_refetch, nocache=nocache)
            if response is not None:
                # Download file
                local_file = open(path, "wb")
                local_file.write(response)
                local_file.close()
                # Finished
                return True
            else:
                return False
        except Exception as e:
            if hasattr(e, 'reason'):
                getLogger().exception('We failed to reach a server. Reason: %s' % (e.reason,))
            elif hasattr(e, 'code'):
                getLogger().exception("The server couldn't fulfill the request. Error code: {code}".format(code=e.code))
            else:
                # everything is fine
                getLogger().exception("Unknown error: %s" % (e,))
        return False