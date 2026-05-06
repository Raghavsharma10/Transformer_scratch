def select_torrent(self):
        """Select torrent.

        First check if specific element/info is obtained in content_page.
        Specify to user if it wants best rated torrent or select one from list.
        If the user wants best rated: Directly obtain magnet/torrent.
        Else: build table with all data and enable the user select the torrent.
        """
        try:
            self.found_torrents = not bool(self.key_search in
                                           self.content_page.text)
            if not self.found_torrents:
                print('No torrents found.')
                sys.exit(1)
            self.soupify()
            if self.mode_search == 'list':
                self.build_table()
                if len(self.hrefs) == 1:
                    print('Press "0" to download it.')
                elif len(self.hrefs) >= 2:
                    print('\nSelect one of the following torrents. ' +
                          'Enter a number between: 0 and ' +
                          str(len(self.hrefs) - 1))

                print('If you want to exit write "' +
                      Colors.LRED + 'Q' + Colors.ENDC + '" or "' +
                      Colors.LRED + 'q' + Colors.ENDC + '".')
                print('If you want to go back to menu and search again write "' +
                      Colors.LGREEN + 'B' + Colors.ENDC + '" or "' +
                      Colors.LGREEN + 'b' + Colors.ENDC + '".')
                while not(self.picked_choice):
                    self.picked_choice = self.handle_select()
        except Exception:
            print('ERROR select_torrent: ')
            logging.error(traceback.format_exc())
            sys.exit(0)