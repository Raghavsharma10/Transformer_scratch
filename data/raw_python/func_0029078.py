def remove_topic(self, topic_id):
        """
        删除话题（需要先删除所有评论，使用默认参数）
        
        :param topic_id: 话题ID
        :return: None
        """
        comment_start = 0
        while comment_start is not None:
            comments = self.list_comments(topic_id, comment_start)
            for comment in comments['results']:
                self.remove_comment(topic_id, comment['id'])
            comment_start = comments['next_start']
        return self.api.req(API_GROUP_REMOVE_TOPIC % topic_id, params={'ck': self.api.ck()})