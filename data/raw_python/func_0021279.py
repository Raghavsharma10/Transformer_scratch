def build_url(self):
        """Build appropiate encoded URL.

        This implies the same way of searching a torrent as in the page itself.
        """
        url = requests.utils.requote_uri(
            self.torrent_page + self.string_search)
        if self.page == '1337x':
            return(url + '/1/')
        elif self.page == 'limetorrents':
            return(url + '/')
        else:
            return(url)