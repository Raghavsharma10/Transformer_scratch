def list_reced_topics(self, user_alias=None, start=0):
        """
        推荐的话题列表
        
        :param user_alias: 指定用户，默认当前
        :param start: 翻页
        :return: 带下一页的列表
        """
        user_alias = user_alias or self.api.user_alias
        xml = self.api.xml(API_GROUP_LIST_USER_RECED_TOPICS % user_alias, params={'start': start})
        return build_list_result(self._parse_topic_table(xml, 'title,comment,time,group,rec'), xml)