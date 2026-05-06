def _execute_search_request(self, url):
        """method to execute the query to google.

        The specified page and keyword
        must already included in the url.
        """
        try:
            self.request_page = requests.get(url)
        except requests.ConnectionError:
            print("Connection to {0} failed".format(str(url)))
        self.current_html_page = self.request_page.text
        soup = BeautifulSoup(self.current_html_page, "html5lib")
        results = soup.find_all('a', class_=False)
        links = []  # store the final url of search result, 10 links
        # this loop filter the search result links inside the search page
        for target in results:
            # filter only link from search result should be appended
            if target.get('href').find("/url?q") == 0 \
                    and not \
                    target.get('href').find(
                            "/url?q=http://webcache.googleusercontent.com"
                    ) == 0 \
                    and not \
                    target.get('href').find("/url?q=/settings/") == 0:
                start_index = target.get('href').find('http')
                end_index = target.get('href').find('&sa')
                # slice the desired url into ideal link, and append
                # it to reserved list variable
                links.append(target.get('href')[start_index:end_index])
        # this loop inspect if the current page is the final page
        for href in results:
            fnl = 'repeat the search with the omitted results included'
            if href.get_text() == fnl:
                self.is_final_page = True
            else:
                pass
        # send the final url link to class reserved variable
        for link in links:
            self.search_result.append(link)