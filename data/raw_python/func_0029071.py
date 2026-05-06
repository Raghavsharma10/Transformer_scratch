def list_topics(self, group_alias, _type='', start=0):
        """
        小组内话题列表
        
        :param group_alias: 小组ID
        :param _type: 类型 默认最新，hot:最热
        :param start: 翻页
        :return: 带下一页的列表
        """
        xml = self.api.xml(API_GROUP_LIST_GROUP_TOPICS % group_alias, params={
            'start': start,
            'type': _type,
        })
        return build_list_result(self._parse_topic_table(xml, 'title,author,comment,updated'), xml)