def fromid(self, item_id):
        """
        Initializes an instance of Story for given item_id.
        It is assumed that the story referenced by item_id is valid
        and does not raise any HTTP errors.
        item_id is an int.
        """
        if not item_id:
            raise Exception('Need an item_id for a story')
        # get details about a particular story
        soup = get_item_soup(item_id)

        # this post has not been scraped, so we explititly get all info
        story_id = item_id
        rank = -1

        # to extract meta information about the post
        info_table = soup.findChildren('table')[2]
        # [0] = title, domain, [1] = points, user, time, comments
        info_rows = info_table.findChildren('tr')

        # title, domain
        title_row = info_rows[0].findChildren('td')[1]
        title = title_row.find('a').text
        try:
            domain = title_row.find('span').string[2:-2]
            # domain found
            is_self = False
            link = title_row.find('a').get('href')
        except AttributeError:
            # self post
            domain = BASE_URL
            is_self = True
            link = '%s/item?id=%s' % (BASE_URL, item_id)

        # points, user, time, comments
        meta_row = info_rows[1].findChildren('td')[1].contents
        # [<span id="score_7024626">789 points</span>, u' by ', <a href="user?id=endianswap">endianswap</a>,
        # u' 8 hours ago  | ', <a href="item?id=7024626">238 comments</a>]

        points = int(re.match(r'^(\d+)\spoint.*', meta_row[0].text).groups()[0])
        submitter = meta_row[2].text
        submitter_profile = '%s/%s' % (BASE_URL, meta_row[2].get('href'))
        published_time = ' '.join(meta_row[3].strip().split()[:3])
        comments_link = '%s/item?id=%s' % (BASE_URL, item_id)
        try:
            num_comments = int(re.match(r'(\d+)\s.*', meta_row[
                4].text).groups()[0])
        except AttributeError:
            num_comments = 0
        story = Story(rank, story_id, title, link, domain, points, submitter,
                      published_time, submitter_profile, num_comments,
                      comments_link, is_self)
        return story