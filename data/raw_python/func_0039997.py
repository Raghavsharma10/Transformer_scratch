def upload(self,
               url,
               method="POST",
               file_path=None):
        """
        :param url:
        :type url: str
        :param method:
        :type method: str
        :param file_path:
        :type file_path: str
        """
        if not os.path.exists(file_path):
            raise RuntimeError("")

        with open_file(file_path, 'rb') as file:
            size = os.path.getsize(file_path)
            label = "Uploading {filename} ({size:.2f}MB)".format(
                filename=os.path.basename(file_path),
                size=size / float(self.chunk_size) / self.chunk_size
            )

            if method == "PUT":
                resp = put(url, data=file)
            elif method == "POST":
                resp = post(url, data=file)

            content_iter = resp.iter_content(chunk_size=self.chunk_size)

            with progressbar(content_iter,
                             length=size / self.chunk_size,
                             label=label) as bar:
                for _ in bar:
                    pass