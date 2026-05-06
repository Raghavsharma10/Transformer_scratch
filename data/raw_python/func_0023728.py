def _get_max_page(self, url):
        """
        Open url and return amount of pages
        """
        html = requests.get(url).text
        pq = PyQuery(html)
        try:
            tds = int(pq("h2").text().split()[-1])
            if tds % 25:
                return tds / 25 + 1
            return tds / 25
        except ValueError:
            raise ValueError("No results found!")