def list_user_topics(self, start=0):
        """
        发表的话题
        
        :param start: 翻页
        :return: 带下一页的列表
        """
        xml = self.api.xml(API_GROUP_LIST_USER_PUBLISHED_TOPICS % self.api.user_alias, params={'start': start})
        return build_list_result(self._parse_topic_table(xml, 'title,comment,created,group'), xml)