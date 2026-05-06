def _parse_topic_table(self, xml, tds='title,created,comment,group', selector='//table[@class="olt"]//tr'):
        """
        解析话题列表
        
        :internal
        :param xml: 页面XML 
        :param tds: 每列的含义，可以是title, created, comment, group, updated, author, time, rec
        :param selector: 表在页面中的位置
        :return: 
        """
        xml_results = xml.xpath(selector)
        results = []
        tds = tds.split(',')
        for item in xml_results:
            try:
                result = {}
                index = 0
                for td in tds:
                    index += 1
                    if td == 'title':
                        xml_title = item.xpath('.//td[position()=%s]/a' % index)[0]
                        url = xml_title.get('href')
                        tid = int(slash_right(url))
                        title = xml_title.text
                        result.update({'id': tid, 'url': url, 'title': title})
                    elif td == 'created':
                        xml_created = item.xpath('.//td[position()=%s]/a' % index) \
                                      or item.xpath('.//td[position()=%s]' % index)
                        created_at = xml_created[0].get('title')
                        result['created_at'] = created_at
                    elif td == 'comment':
                        xml_comment = item.xpath('.//td[position()=%s]/span' % index) \
                                      or item.xpath('.//td[position()=%s]' % index)
                        comment_count = int(re.match(r'\d+', xml_comment[0].text).group())
                        result['comment_count'] = comment_count
                    elif td == 'group':
                        xml_group = item.xpath('.//td[position()=%s]/a' % index)[0]
                        group_url = xml_group.get('href')
                        group_alias = slash_right(group_url)
                        group_name = xml_group.text
                        result.update({'group_alias': group_alias, 'group_url': group_url, 'group_name': group_name})
                    elif td == 'author':
                        xml_author = item.xpath('.//td[position()=%s]/a' % index)[0]
                        author_url = xml_author.get('href')
                        author_alias = slash_right(author_url)
                        author_nickname = xml_author.text
                        result.update({
                            'author_url': author_url,
                            'author_alias': author_alias,
                            'author_nickname': author_nickname,
                        })
                    elif td == 'updated':
                        result['updated_at'] = item.xpath('.//td[position()=%s]/text()' % index)[0]
                    elif td == 'time':
                        result['time'] = item.xpath('.//td[position()=%s]/text()' % index)[0]
                    elif td == 'rec':
                        xml_rec = item.xpath('.//td[position()=%s]//a[@class="lnk-remove"]/@href' % (index - 1))[0]
                        result['rec_id'] = re.search(r'rec_id=(\d+)', xml_rec).groups()[0]
                results.append(result)
            except Exception as e:
                self.api.api.logger.exception('parse topic table exception: %s' % e)
        return results