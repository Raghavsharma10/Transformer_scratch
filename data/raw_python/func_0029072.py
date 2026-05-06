def list_joined_topics(self, start=0):
        """
        已加入的所有小组的话题列表
        
        :param start: 翻页
        :return: 带下一页的列表
        """
        xml = self.api.xml(API_GROUP_HOME, params={'start': start})
        return build_list_result(self._parse_topic_table(xml, 'title,comment,created,group'), xml)