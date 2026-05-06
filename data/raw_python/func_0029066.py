def search_groups(self, keyword, start=0):
        """
        搜索小组
        
        :param keyword: 搜索的关键字
        :param start: 翻页
        :return: 含总数的列表
        """
        xml = self.api.xml(API_GROUP_SEARCH_GROUPS % (start, keyword))
        xml_results = xml.xpath('//div[@class="groups"]/div[@class="result"]')
        results = []
        for item in xml_results:
            try:
                url = item.xpath('.//h3/a/@href')[0]
                info = item.xpath('.//div[@class="content"]/div[@class="info"]/text()')[0].strip(' ')
                onclick = item.xpath('.//h3/a/@onclick')[0]
                meta = {
                    'icon': item.xpath('.//img/@src')[0],
                    'id': re.search(r'sid[^\d]+(\d+)', onclick).groups()[0],
                    'url': url,
                    'alias': url.rstrip('/').rsplit('/', 1)[1],
                    'name': item.xpath('.//h3/a/text()')[0],
                    'user_count': int(re.match(r'\d+', info).group()),
                    'user_alias': re.search(r'个(.+)\s*在此', info).groups()[0],
                }
                results.append(meta)
            except Exception as e:
                self.api.logger.exception('parse search groups result error: %s' % e)
        return build_list_result(results, xml)