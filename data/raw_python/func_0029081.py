def add_comment(self, topic_id, content, reply_id=None):
        """
        添加评论
        
        :param topic_id: 话题ID
        :param content: 内容
        :param reply_id: 回复ID
        :return: None
        """
        return self.api.req(API_GROUP_ADD_COMMENT % topic_id, 'post', data={
            'ck': self.api.ck(),
            'ref_cid': reply_id,
            'rv_comment': content,
            'start': 0,
            'submit_btn': '加上去',
        })