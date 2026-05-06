def download_as_file(self, url: str, folder: Path, name: str, delay: float = 0) -> str:
        """
        Download the given url to the given target folder.

        :param url: link
        :type url: str
        :param folder: target folder
        :type folder: ~pathlib.Path
        :param name: target file name
        :type name: str
        :param delay: after download wait in seconds
        :type delay: float
        :return: url
        :rtype: str
        :raises ~urllib3.exceptions.HTTPError: if the connection has an error
        """
        while folder.joinpath(name).exists():  # TODO: handle already existing files
            self.log.warning('already exists: ' + name)
            name = name + '_d'

        with self._downloader.request('GET', url, preload_content=False, retries=urllib3.util.retry.Retry(3)) as reader:
            if reader.status == 200:
                with folder.joinpath(name).open(mode='wb') as out_file:
                    out_file.write(reader.data)
            else:
                raise HTTPError(f"{url} | {reader.status}")

        if delay > 0:
            time.sleep(delay)

        return url