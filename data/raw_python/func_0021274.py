def download_torrent(self):
        """Download torrent.

        Rated implies download the unique best rated torrent found.
        Otherwise: get the magnet and download it.
        """
        try:
            if self.back_to_menu is True:
                return
            if self.found_torrents is False:
                print('Nothing found.')
                return
            if self.mode_search == 'best_rated':
                print('Downloading..')
                self.open_magnet()
            elif self.mode_search == 'list':
                if self.selected is not None:
                    # t_p, pirate and 1337x got magnet inside, else direct.
                    if self.page in ['eztv', 'limetorrents']:
                        self.magnet = self.hrefs[int(self.selected)]
                        print('Downloading..')
                        self.open_magnet()
                    elif self.page in ['the_pirate_bay',
                                       'torrent_project',
                                       '1337x',
                                       'isohunt']:
                        url = self.hrefs[int(self.selected)]
                        self.get_magnet(url)
                        print('Downloading..')
                        self.open_magnet()
                    else:
                        print('Bad selected page.')
                else:
                    print('Nothing selected.')
                    sys.exit(1)
        except Exception:
            print(traceback.format_exc())
            sys.exit(0)