def _build_comments(self, soup):
        """
        For the story, builds and returns a list of Comment objects.
        """

        comments = []
        current_page = 1

        while True:
            # Get the table holding all comments:
            if current_page == 1:
                table = soup.findChildren('table')[3]
            elif current_page > 1:
                table = soup.findChildren('table')[2]
            # get all rows (each comment is duplicated twice)
            rows = table.findChildren(['tr'])
            # last row is more, second last is spacing
            rows = rows[:len(rows) - 2]
            # now we have unique comments only
            rows = [row for i, row in enumerate(rows) if (i % 2 == 0)]

            if len(rows) > 1:
                for row in rows:

                    # skip an empty td
                    if not row.findChildren('td'):
                        continue

                    # Builds a flat list of comments

                    # level of comment, starting with 0
                    level = int(row.findChildren('td')[1].find('img').get(
                        'width')) // 40

                    spans = row.findChildren('td')[3].findAll('span')
                    # span[0] = submitter details
                    # [<a href="user?id=jonknee">jonknee</a>, u' 1 hour ago  | ', <a href="item?id=6910978">link</a>]
                    # span[1] = actual comment

                    if str(spans[0]) != '<span class="comhead"></span>':
                        # user who submitted the comment
                        user = spans[0].contents[0].string
                        # relative time of comment
                        time_ago = spans[0].contents[1].string.strip(
                        ).rstrip(' |')
                        try:
                            comment_id = int(re.match(r'item\?id=(.*)',
                                                      spans[0].contents[
                                                          2].get(
                                                          'href')).groups()[0])
                        except AttributeError:
                            comment_id = int(re.match(r'%s/item\?id=(.*)' %
                                                      BASE_URL,
                                                      spans[0].contents[
                                                          2].get(
                                                          'href')).groups()[0])

                        # text representation of comment (unformatted)
                        body = spans[1].text

                        if body[-2:] == '--':
                            body = body[:-5]

                        # html of comment, may not be valid
                        try:
                            pat = re.compile(
                                r'<span class="comment"><font color=".*">(.*)</font></span>')
                            body_html = re.match(pat, str(spans[1]).replace(
                                '\n', '')).groups()[0]
                        except AttributeError:
                            pat = re.compile(
                                r'<span class="comment"><font color=".*">(.*)</font></p><p><font size="1">')
                            body_html = re.match(pat, str(spans[1]).replace(
                                '\n', '')).groups()[0]

                    else:
                        # comment deleted
                        user = ''
                        time_ago = ''
                        comment_id = -1
                        body = '[deleted]'
                        body_html = '[deleted]'

                    comment = Comment(comment_id, level, user, time_ago,
                                      body, body_html)
                    comments.append(comment)

            # Move on to the next page of comments, or exit the loop if there
            # is no next page.
            next_page_url = self._get_next_page(soup, current_page)
            if not next_page_url:
                break

            soup = get_soup(page=next_page_url)
            current_page += 1

        previous_comment = None
        # for comment in comments:
        # if comment.level == 0:
        #         previous_comment = comment
        #     else:
        #         level_difference = comment.level - previous_comment.level
        #         previous_comment.body_html += '\n' + '\t' * level_difference \
        #                                       + comment.body_html
        #         previous_comment.body += '\n' + '\t' * level_difference + \
        #                                  comment.body
        return comments