def list_joined_groups(self, user_alias=None):
        """
        已加入的小组列表
        
        :param user_alias: 用户名，默认为当前用户名
        :return: 单页列表
        """
        xml = self.api.xml(API_GROUP_LIST_JOINED_GROUPS % (user_alias or self.api.user_alias))
        xml_results = xml.xpath('//div[@class="group-list group-cards"]/ul/li')
        results = []
        for item in xml_results:
            try:
                icon = item.xpath('.//img/@src')[0]
                link = item.xpath('.//div[@class="title"]/a')[0]
                url = link.get('href')
                name = link.text
                alias = url.rstrip('/').rsplit('/', 1)[1]
                user_count = int(item.xpath('.//span[@class="num"]/text()')[0][1:-1])
                results.append({
                    'icon': icon,
                    'alias': alias,
                    'url': url,
                    'name': name,
                    'user_count': user_count,
                })
            except Exception as e:
                self.api.logger.exception('parse joined groups exception: %s' % e)
        return build_list_result(results, xml)