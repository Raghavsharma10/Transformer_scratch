def load(self):
        """Load the lyrics from MetroLyrics."""
        page = requests.get(self._url)
        # Forces utf-8 to prevent character mangling
        page.encoding = 'utf-8'

        tree = html.fromstring(page.text)
        lyric_div = tree.get_element_by_id('lyrics-body-text')
        verses = [c.text_content() for c in lyric_div.find_class('verse')]
        self._lyrics = '\n\n'.join(verses)
        return self