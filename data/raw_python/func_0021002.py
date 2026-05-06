def download_file(self, file_id, dest_file_path,
                            progress_callback=None,
                            chunk_size=1024*1024*1):
        """Download a file.

        The whole file is never loaded in memory.

        The callback(transferred, total) to let you know the download progress.
        Download can be cancelled if the callback raise an Exception.

        >>> def progress_callback(transferred, total):
        ...    print 'Downloaded %i bytes of %i' % (transferred, total, )
        ...    if user_request_cancel:
        ...       raise MyCustomCancelException()

        Args:
            file_id (int): ID of the file to download.

            dest_file_path (str): Local path where to store the downloaded filed.

            progress_callback (func): Function called each time a chunk is downloaded.

            chunk_size (int): Size of chunks.

        Raises:
            BoxError: An error response is returned from Box (status_code >= 400).

            BoxHttpResponseError: Response from Box is malformed.

            requests.exceptions.*: Any connection related problem.
        """
        with open(dest_file_path, 'wb') as fp:
            req = self.__request("GET", "files/%s/content" % (file_id, ),
                                                stream=True,
                                                json_data=False)
            total = -1
            if hasattr(req, 'headers'):
                lower_headers = {k.lower():v for k,v in req.headers.items()}
                if 'content-length' in lower_headers:
                    total = lower_headers['content-length']

            transferred = 0
            for chunk in req.iter_content(chunk_size=chunk_size):
                if chunk: # filter out keep-alive new chunks
                    if progress_callback:
                        progress_callback(transferred, total)
                    fp.write(chunk)
                    fp.flush()
                    transferred += len(chunk)

            if progress_callback:
                progress_callback(transferred, total)