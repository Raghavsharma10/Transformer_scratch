def remove_comment(self, topic_id, comment_id, reason='0', other=None):
        """
        删除评论（自己发的话题所有的都可以删除，否则只能删自己发的）
        
        :param topic_id: 话题ID
        :param comment_id: 评论ID
        :param reason: 原因 0/1/2 （内容不符/反动/其它）
        :param other: 其它原因的具体(2)
        :return: None
        """
        params = {'cid': comment_id}
        data = {'cid': comment_id, 'ck': self.api.ck(), 'reason': reason, 'other': other, 'submit': '确定'}
        r = self.api.req(API_GROUP_REMOVE_COMMENT % topic_id, 'post', params, data)
        if r.text.find('douban_admin') > -1:
            r = self.api.req(API_GROUP_ADMIN_REMOVE_COMMENT % topic_id, 'post', params, data)
        self.api.logger.debug('remove comment final url is <%s>' % r.url)
        return r