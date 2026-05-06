def download(self,
                 url,
                 dest_path=None):
        """
        :param url:
        :type url: str
        :param dest_path:
        :type dest_path: str
        """
        if os.path.exists(dest_path):
            os.remove(dest_path)

        resp = get(url, stream=True)
        size = int(resp.headers.get("content-length"))
        label = "Downloading {filename} ({size:.2f}MB)".format(
            filename=os.path.basename(dest_path),
            size=size / float(self.chunk_size) / self.chunk_size
        )

        with open_file(dest_path, 'wb') as file:
            content_iter = resp.iter_content(chunk_size=self.chunk_size)
            with progressbar(content_iter,
                             length=size / self.chunk_size,
                             label=label) as bar:
                for chunk in bar:
                    if chunk:
                        file.write(chunk)
                        file.flush()