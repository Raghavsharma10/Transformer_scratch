def visit(self, url=''):
        """
        Driver gets the provided url in the browser, returns True if successful

        url -- An absolute or relative url stored as a string
        """
        def _visit(url):
            if len(url) > 0 and url[0] == '/':
                # url's first character is a forward slash; treat as relative path
                path = url
                full_url = self.driver.current_url
                parsed_url = urlparse(full_url)
                base_url = str(parsed_url.scheme) + '://' + str(parsed_url.netloc)
                url = urljoin(base_url, path)

            try:
                return self.driver.get(url)
            except TimeoutException:
                if self.ignore_page_load_timeouts:
                    pass
                else:
                    raise PageTimeoutException.PageTimeoutException(self, url)

        return self.execute_and_handle_webdriver_exceptions(lambda: _visit(url))