def build_table(self):
        """Build table."""
        headers = ['Title', 'Seeders', 'Leechers', 'Age', 'Size']
        titles = []
        seeders = []
        leechers = []
        ages = []
        sizes = []

        if self.page == 'torrent_project':
            titles = [list(span.find('a').stripped_strings)[0]
                      for span in self.elements[0]]

            seeders = [span.get_text() for span in self.elements[1]]
            leechers = [span.get_text() for span in self.elements[2]]
            ages = [span.get_text() for span in self.elements[3]]
            sizes = [span.get_text() for span in self.elements[4]]

            # Torrents
            self.hrefs = [self.domain +
                          span.find('a')['href']
                          for span in self.elements[0]]

        elif self.page == 'the_pirate_bay':
            for elem in self.elements[0]:
                title = elem.find('a', {'class': 'detLink'}).get_text()
                titles.append(title)

                font_text = elem.find(
                    'font', {'class': 'detDesc'}).get_text()
                dammit = UnicodeDammit(font_text)
                age, size = dammit.unicode_markup.split(',')[:-1]
                ages.append(age)
                sizes.append(size)
                # Torrent
                href = self.domain + \
                    elem.find('a', title=re.compile('magnet'))['href']
                self.hrefs.append(str(href))

            seeders = [elem.get_text() for elem in self.elements[1]]
            leechers = [elem.get_text() for elem in self.elements[2]]

        elif self.page == '1337x':
            titles = [elem.get_text() for elem in self.elements[0]]
            seeders = [elem.get_text() for elem in self.elements[1]]
            leechers = [elem.get_text() for elem in self.elements[2]]
            ages = [elem.get_text() for elem in self.elements[3]]
            sizes = [elem.get_text('|').split('|')[0]
                     for elem in self.elements[4]]

            # Torrent
            self.hrefs = [self.domain +
                          elem.find(href=re.compile('torrent'))['href']
                          for elem in self.elements[0]]

        elif self.page == 'eztv':
            titles = [elem.get_text() for elem in self.elements[0]]
            seeders = [elem.get_text() for elem in self.elements[4]]
            leechers = ['-' for elem in self.elements[4]]
            ages = [elem.get_text() for elem in self.elements[3]]
            sizes = [elem.get_text() for elem in self.elements[2]]

            # Magnets
            self.hrefs = [elem.find(href=re.compile('magnet'))['href']
                          for elem in self.elements[1]]

        elif self.page == 'limetorrents':
            titles = [elem.get_text() for elem in self.elements[0]]
            seeders = [elem.get_text() for elem in self.elements[3]]
            leechers = [elem.get_text() for elem in self.elements[4]]
            ages = [elem.get_text() for elem in self.elements[1]]
            sizes = [elem.get_text() for elem in self.elements[2]]

            # Magnets
            self.hrefs = [elem.find('a', href=re.compile('torrent'))['href']
                          for elem in self.elements[0]]

        elif self.page == 'isohunt':
            titles = [elem.get_text() for elem in self.elements[0]]
            seeders = [elem.get_text() for elem in self.elements[5]]
            leechers = ['-' for elem in self.elements[5]]
            ages = [elem.get_text() for elem in self.elements[3]]
            sizes = [elem.get_text() for elem in self.elements[4]]

            # Torrents
            self.hrefs = [self.domain +
                          elem.find(href=re.compile('torrent_details'))['href']
                          for elem in self.elements[0]]
        else:
            print('Error page')

        self.table = [[Colors.BOLD +
                       UnicodeDammit(titles[i][:75].strip(), ["utf-8"]).unicode_markup +
                       Colors.ENDC
                       if (i + 1) % 2 == 0
                       else UnicodeDammit(
                           titles[i][:75].strip()).unicode_markup,
                       Colors.SEEDER + seeders[i].strip() + Colors.ENDC
                       if (i + 1) % 2 == 0
                       else Colors.LGREEN + seeders[i].strip() + Colors.ENDC,
                       Colors.LEECHER + leechers[i].strip() + Colors.ENDC
                       if (i + 1) % 2 == 0
                       else Colors.LRED + leechers[i].strip() + Colors.ENDC,
                       Colors.LIGHTBLUE + ages[i].strip() + Colors.ENDC
                       if (i + 1) % 2 == 0
                       else Colors.BLUE + ages[i].strip() + Colors.ENDC,
                       Colors.PINK + sizes[i].strip() + Colors.ENDC
                       if (i + 1) % 2 == 0
                       else Colors.PURPLE + sizes[i].strip() + Colors.ENDC]
                      for i in range(len(self.hrefs))]

        print(tabulate(self.table,
                       headers=headers,
                       tablefmt='psql',
                       numalign='right',
                       stralign='left',
                       showindex=True))