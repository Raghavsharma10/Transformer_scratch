def search_topics(self, keyword, sort='relevance', start=0):
        """
        搜索话题
        
        :param keyword: 关键字
        :param sort: 排序方式 relevance/newest
        :param start: 翻页
        :return: 带总数的列表
        """
        xml = self.api.xml(API_GROUP_SEARCH_TOPICS % (start, sort, keyword))
        return build_list_result(self._parse_topic_table(xml), xml)