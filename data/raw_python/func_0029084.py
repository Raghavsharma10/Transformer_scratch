def remove_commented_topic(self, topic_id):
        """
        删除回复的话题（删除所有自己发布的评论）
        
        :param topic_id: 话题ID
        :return: None
        """
        return [self.remove_comment(topic_id, item['id']) for item in self.list_user_comments(topic_id)]