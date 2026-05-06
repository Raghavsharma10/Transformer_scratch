def list_comments(self, topic_id, start=0):
        """
        回复列表
        
        :param topic_id: 话题ID
        :param start: 翻页
        :return: 带下一页的列表
        """
        xml = self.api.xml(API_GROUP_GET_TOPIC % topic_id, params={'start': start})
        xml_results = xml.xpath('//ul[@id="comments"]/li')
        results = []
        for item in xml_results:
            try:
                author_avatar = item.xpath('.//img/@src')[0]
                author_url = item.xpath('.//div[@class="user-face"]/a/@href')[0]
                author_alias = slash_right(author_url)
                author_signature = item.xpath('.//h4/text()')[1].strip()
                author_nickname = item.xpath('.//h4/a/text()')[0].strip()
                created_at = item.xpath('.//h4/span/text()')[0].strip()
                content = etree.tostring(item.xpath('.//div[@class="reply-doc content"]/p')[0]).decode('utf8').strip()
                cid = item.get('id')
                results.append({
                    'id': cid,
                    'author_avatar': author_avatar,
                    'author_url': author_url,
                    'author_alias': author_alias,
                    'author_signature': author_signature,
                    'author_nickname': author_nickname,
                    'created_at': created_at,
                    'content': unescape(content),
                })
            except Exception as e:
                self.api.logger.exception('parse comment exception: %s' % e)
        return build_list_result(results, xml)