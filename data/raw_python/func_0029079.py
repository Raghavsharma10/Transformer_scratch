def update_topic(self, topic_id, title, content):
        """
        更新话题
        
        :param topic_id: 话题ID
        :param title: 标题
        :param content: 内容
        :return: bool
        """
        xml = self.api.req(API_GROUP_UPDATE_TOPIC % topic_id, 'post', data={
            'ck': self.api.ck(),
            'rev_title': title,
            'rev_text': content,
            'rev_submit': '好了，改吧',
        })
        return not xml.url.startswith(API_GROUP_UPDATE_TOPIC % topic_id)