def soupify(self):
        """Get proper torrent/magnet information.

        If search_mode is rated then get torrent/magnet.
        If not, get all the elements to build the table.
        There are different ways for each page.
        """
        soup = BeautifulSoup(self.content_page.content, 'lxml')
        if self.page == 'torrent_project':
            main = soup.find('div', {'id': 'similarfiles'})
            if self.mode_search == 'best_rated':
                rated_url = self.domain + \
                    main.find(href=re.compile('torrent.html'))['href']
                self.get_magnet(rated_url)
            else:
                divs = main.find_all('div', limit=30)[2:]
                self.elements = list(
                    zip(*[d.find_all('span', recursive=False)
                          for d in divs]))  # Torrents

        elif self.page == 'the_pirate_bay':
            main = soup.find('table', {'id': 'searchResult'})
            if self.mode_search == 'best_rated':
                rated_url = self.domain + \
                    main.find('a', href=re.compile('torrent'))['href']
                self.get_magnet(rated_url)
            else:
                trs = main.find_all('tr', limit=30)[1:]
                self.elements = list(
                    zip(*[tr.find_all('td', recursive=False)[1:]
                          for tr in trs]))  # Magnets

        elif self.page == '1337x':
            main = soup.find('table', {'class': 'table'})
            if self.mode_search == 'best_rated':
                rated_url = self.domain + \
                    main.find('a', href=re.compile('torrent'))['href']
                self.get_magnet(rated_url)
            else:
                trs = main.find_all('tr', limit=30)[1:]
                self.elements = list(
                    zip(*([tr.find_all('td', recursive=False)[:-1]
                           for tr in trs])))  # Torrents

        elif self.page == 'eztv':
            main = soup.find_all('table', {'class': 'forum_header_border'})[2]
            if self.mode_search == 'best_rated':
                self.magnet = main.find('a', href=re.compile('magnet'))['href']
            else:
                trs = main.find_all('tr', limit=30)[2:]
                self.elements = list(
                    zip(*([tr.find_all('td', recursive=False)[1:-1]
                           for tr in trs])))  # Magnets

        elif self.page == 'limetorrents':
            main = soup.find('table', {'class': 'table2'})
            if self.mode_search == 'best_rated':
                self.magnet = main.find(
                    'a', href=re.compile('torrent'))['href']
            else:
                trs = main.find_all('tr', limit=30)[1:]
                self.elements = list(
                    zip(*([tr.find_all('td', recursive=False)[:-1]
                           for tr in trs])))  # Magnets

        elif self.page == 'isohunt':
            main = soup.find('table', {'class': 'table'})
            if self.mode_search == 'best_rated':
                rated_url = self.domain + \
                    main.find('a', href=re.compile(
                        'torrent_details'))['href']
                self.get_magnet(rated_url)
            else:
                trs = main.find_all('tr', limit=30)[1:-1]
                self.elements = list(
                    zip(*([tr.find_all('td', recursive=False)[1:-1]
                           for tr in trs])))  # Torrent
        else:
            print('Cannot soupify current page. Try again.')