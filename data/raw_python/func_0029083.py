def list_user_comments(self, topic_id, user_alias=None):
        """
        列出用户在话题下的所有回复
        
        :param topic_id: 话题ID
        :param user_alias: 用户ID，默认当前
        :return: 纯列表
        """
        user_alias = user_alias or self.api.user_alias
        comment_start = 0
        results = []
        while comment_start is not None:
            comments = self.list_comments(topic_id, comment_start)
            results += [item for item in comments['results'] if item['author_alias'] == user_alias]
            comment_start = comments['next_start']
        return results