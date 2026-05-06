def download(self, *ids):
        """
        Downloads the subtitles with the given ids.
        :param ids: The subtitles to download
        :return: Result instances
        :raises NotOKException
        """
        bundles = sublists_of(ids, 20)  # 20 files at once is an API restriction

        for bundle in bundles:
            download_response = self._rpc.DownloadSubtitles(self._token, bundle)

            assert_status(download_response)

            download_data = download_response.get('data')

            for item in download_data:
                subtitle_id = item['idsubtitlefile']
                subtitle_data = item['data']

                decompressed = decompress(subtitle_data)

                yield Result(subtitle_id, decompressed)