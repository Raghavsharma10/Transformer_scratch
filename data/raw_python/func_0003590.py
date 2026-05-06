def create_chapter_from_url(self, url, title=None):
        """
        Creates a Chapter object from a url. Pulls the webpage from the
        given url, sanitizes it using the clean_function method, and saves
        it as the content of the created chapter. Basic webpage loaded
        before any javascript executed.

        Args:
            url (string): The url to pull the content of the created Chapter
                from
            title (Option[string]): The title of the created Chapter. By
                default, this is None, in which case the title will try to be
                inferred from the webpage at the url.

        Returns:
            Chapter: A chapter object whose content is the webpage at the given
                url and whose title is that provided or inferred from the url

        Raises:
            ValueError: Raised if unable to connect to url supplied
        """
        try:
            request_object = requests.get(url, headers=self.request_headers, allow_redirects=False)
        except (requests.exceptions.MissingSchema,
                requests.exceptions.ConnectionError):
            raise ValueError("%s is an invalid url or no network connection" % url)
        except requests.exceptions.SSLError:
            raise ValueError("Url %s doesn't have valid SSL certificate" % url)
        unicode_string = request_object.text
        return self.create_chapter_from_string(unicode_string, url, title)