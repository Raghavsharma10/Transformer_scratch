def __readAPI(self, url):
        """Read a petition to the GitHub API (private).

        :param url: URL to query.
        :type url: str.
        :return: the response of the API -a dictionary with
            these fields-:
            * total_count (int): number of total
            users that match with the search
            * incomplete_results (bool):
            https://developer.github.com/v3/search/#timeouts-and-incomplete-results
            * items (List[dict]): a list with the
            users that match with the search
        :rtype: dict.
        """
        code = 0
        hdr = {'User-Agent': 'curl/7.43.0 (x86_64-ubuntu) \
               libcurl/7.43.0 OpenSSL/1.0.1k zlib/1.2.8 gh-rankings-grx',
               'Accept': 'application/vnd.github.v3.text-match+json',
               'Accept-Encoding': 'gzip'}
        while code != 200:
            req = Request(url, headers=hdr)
            try:
                self.__logger.debug("Getting " + url)
                response = urlopen(req)
                code = response.code
            except HTTPError as error:
                if error.code == 404:
                    self.__logger.exception("_readAPI: ERROR 404")
                    self.__logger.exception(str(error))
                    break
                headers = error.headers.items()
                reset = -1
                for header in headers:
                    if header[0] == "X-RateLimit-Reset":
                        reset = int(header[1])
                if reset < 0:
                    log_message = "Error when reading response. Wait: 30 secs"
                    sleep_duration = 30
                else:
                    utcAux = datetime.datetime.utcnow()
                    utcAux = utcAux.utctimetuple()
                    now_sec = timegm(utcAux)
                    sleep_duration = reset - now_sec
                    log_message = "Limit of API. Wait: "
                    log_message += str(sleep_duration)
                    log_message += " secs"
                self.__logger.warning(log_message)
                sleep(sleep_duration)
                code = 0
            except URLError as error:
                self.__logger.exception(str(error))
                self.__logger.exception("_readAPI: waiting 15 secs")
                sleep(15)
        responseBody = response.read()

        if response.getheader('Content-Encoding') == 'gzip':
            with GzipFile(fileobj=BytesIO(responseBody)) as gzFile:
                responseBody = gzFile.read()

        data = loads(responseBody.decode('utf-8'))
        response.close()
        return data