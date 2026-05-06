def load(self, verbose=False):
        """
        Load the list of songs.

        Note that this only loads a list of songs that this artist was the main
        artist of.  If they were only featured in the song, that song won't be
        listed here.  There is a list on the artist page for that, I just
        haven't added any parsing code for that, since I don't need it.
        """
        self._songs = []
        page_num = 1
        total_pages = 1

        while page_num <= total_pages:
            if verbose:
                print('retrieving page %d' % page_num)
            page = requests.get(ARTIST_URL.format(artist=self.name,
                                                  n=page_num))
            tree = html.fromstring(page.text)
            song_rows_xp = r'//*[@id="popular"]/div/table/tbody/tr'
            songlist_pagination_xp = r'//*[@id="main-content"]/div[1]/'\
                                     'div[2]/p/span/a'

            rows = tree.xpath(song_rows_xp)
            for row in rows:
                song_link = row.xpath(r'./td/a[contains(@class,"title")]')
                assert len(song_link) == 1
                self._songs.append(Song(url=song_link[0].attrib['href']))

            total_pages = len(tree.xpath(songlist_pagination_xp))
            page_num += 1
        return self