def get_magnet(self, url):
        """Get magnet from torrent page. Url already got domain."""
        content_most_rated = requests.get(url)
        rated_soup = BeautifulSoup(content_most_rated.content, 'lxml')

        if self.page == 'torrent_project':
            self.magnet = rated_soup.find(
                'a', href=True, text=re.compile('Download'))['href']

        elif self.page == 'the_pirate_bay':
            self.magnet = rated_soup.find(
                'a', href=True, text=re.compile('Get this torrent'))['href']

        elif self.page == '1337x':
            div1337 = rated_soup.find(
                'div', {'class': 'torrent-category-detail'})
            self.magnet = div1337.find('a', href=re.compile('magnet'))['href']

        elif self.page == 'isohunt':
            self.magnet = rated_soup.find(
                'a', href=re.compile('magnet'))['href']

        else:
            print('Wrong page to get magnet!')
            sys.exit(1)