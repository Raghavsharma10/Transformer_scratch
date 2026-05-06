def _build_story(self, all_rows):
        """
        Builds and returns a list of stories (dicts) from the passed source.
        """
        # list to hold all stories
        all_stories = []

        for (info, detail) in all_rows:

            #-- Get the into about a story --#
            # split in 3 cells
            info_cells = info.findAll('td')

            rank = int(info_cells[0].string[:-1])
            title = '%s' % info_cells[2].find('a').string
            link = info_cells[2].find('a').get('href')

            # by default all stories are linking posts
            is_self = False

            # the link doesn't contains "http" meaning an internal link
            if link.find('item?id=') is -1:
                # slice " (abc.com) "
                domain = info_cells[2].findAll('span')[1].string[2:-1]
            else:
                link = '%s/%s' % (BASE_URL, link)
                domain = BASE_URL
                is_self = True
            #-- Get the into about a story --#

            #-- Get the detail about a story --#
            # split in 2 cells, we need only second
            detail_cell = detail.findAll('td')[1]
            # list of details we need, 5 count
            detail_concern = detail_cell.contents

            num_comments = -1

            if re.match(r'^(\d+)\spoint.*', detail_concern[0].string) is not \
                    None:
                # can be a link or self post
                points = int(re.match(r'^(\d+)\spoint.*', detail_concern[
                    0].string).groups()[0])
                submitter = '%s' % detail_concern[2].string
                submitter_profile = '%s/%s' % (BASE_URL, detail_concern[
                    2].get('href'))
                published_time = ' '.join(detail_concern[3].strip().split()[
                                          :3])
                comment_tag = detail_concern[4]
                story_id = int(re.match(r'.*=(\d+)', comment_tag.get(
                    'href')).groups()[0])
                comments_link = '%s/item?id=%d' % (BASE_URL, story_id)
                comment_count = re.match(r'(\d+)\s.*', comment_tag.string)
                try:
                    # regex matched, cast to int
                    num_comments = int(comment_count.groups()[0])
                except AttributeError:
                    # did not match, assign 0
                    num_comments = 0
            else:
                # this is a job post
                points = 0
                submitter = ''
                submitter_profile = ''
                published_time = '%s' % detail_concern[0]
                comment_tag = ''
                try:
                    story_id = int(re.match(r'.*=(\d+)', link).groups()[0])
                except AttributeError:
                    # job listing that points to external link
                    story_id = -1
                comments_link = ''
                comment_count = -1
            #-- Get the detail about a story --#

            story = Story(rank, story_id, title, link, domain, points,
                          submitter, published_time, submitter_profile,
                          num_comments, comments_link, is_self)

            all_stories.append(story)

        return all_stories