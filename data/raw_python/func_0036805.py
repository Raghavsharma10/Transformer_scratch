def search(self, query: 're.Pattern') -> 'Iterable[_WorkTitles]':
        """Search titles using a compiled RE query."""
        titles: 'Titles'
        for titles in self._titles_list:
            title: 'AnimeTitle'
            for title in titles.titles:
                if query.search(title.title):
                    yield WorkTitles(
                        aid=titles.aid,
                        main_title=_get_main_title(titles.titles),
                        titles=[t.title for t in titles.titles],
                    )
                    continue