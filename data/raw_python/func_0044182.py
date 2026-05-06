def extract(self, html):
        "Extract http-equiv refresh url to follow."
        extracted = {}
        soup = BeautifulSoup(html, parser)
        for meta_tag in soup.find_all('meta'):
            if self.key_attr in meta_tag.attrs and 'content' in meta_tag.attrs and \
                meta_tag[self.key_attr].lower() == self.val_attr:
                refresh = meta_tag.attrs['content']
                try:
                    pause, newurl = self.parse_refresh_header(refresh)
                    if newurl:
                        extracted['urls'] = [newurl]
                        break # one is enough
                except:
                    pass # nevermind
        return extracted