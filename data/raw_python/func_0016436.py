def get_stories(self, story_type='', limit=30):
        """
        Yields a list of stories from the passed page
        of HN.
        'story_type' can be:
        \t'' = top stories (homepage) (default)
        \t'news2' = page 2 of top stories
        \t'newest' = most recent stories
        \t'best' = best stories

        'limit' is the number of stories required from the given page.
        Defaults to 30. Cannot be more than 30.
        """
        if limit is None or limit < 1 or limit > 30:
            # we need at least 30 items
            limit = 30

        stories_found = 0
        # self.more = story_type
        # while we still have more stories to find
        while stories_found < limit:
            # get current page soup
            soup = get_soup(page=story_type)
            all_rows = self._get_zipped_rows(soup)
            # get a list of stories on current page
            stories = self._build_story(all_rows)
            # move to next page
            # self.more = self._get_next_page(soup)

            for story in stories:
                yield story
                stories_found += 1

                # if enough stories found, return
                if stories_found == limit:
                    return