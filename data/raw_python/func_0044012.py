def __getDataFromURL(url):
        """Read HTML data from an user GitHub profile.

        :param url: URL of the webpage to download.
        :type url: str.
        :return: webpage donwloaded.
        :rtype: str.
        """
        code = 0

        while code != 200:
            req = Request(url)
            try:
                response = urlopen(req)
                code = response.code
                sleep(0.01)
            except HTTPError as error:
                code = error.code
                if code == 404:
                    break
            except URLError as error:
                sleep(3)

        if code == 404:
            raise Exception("User was not found")
        return response.read().decode('utf-8')