def list_commented_topics(self, start=0):
        """
        回复过的话题列表
        
        :param start: 翻页
        :return: 带下一页的列表
        """
        xml = self.api.xml(API_GROUP_LIST_USER_COMMENTED_TOPICS % self.api.user_alias, params={'start': start})
        return build_list_result(self._parse_topic_table(xml, 'title,comment,time,group'), xml)