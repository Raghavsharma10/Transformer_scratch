def add_topic(self, group_alias, title, content):
        """
        创建话题（小心验证码~）
        
        :param group_alias: 小组ID
        :param title: 标题
        :param content: 内容
        :return: bool
        """
        xml = self.api.req(API_GROUP_ADD_TOPIC % group_alias, 'post', data={
            'ck': self.api.ck(),
            'rev_title': title,
            'rev_text': content,
            'rev_submit': '好了，发言',
        })
        return not xml.url.startswith(API_GROUP_ADD_TOPIC % group_alias)