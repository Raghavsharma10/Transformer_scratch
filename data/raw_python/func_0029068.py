def join_group(self, group_alias, message=None):
        """
        加入小组
        
        :param group_alias: 小组ID
        :param message: 如果要验证，留言信息
        :return: 枚举
                - joined: 加入成功
                - waiting: 等待审核
                - initial: 加入失败
        """
        xml = self.api.xml(API_GROUP_GROUP_HOME % group_alias, params={
            'action': 'join',
            'ck': self.api.ck(),
        })
        misc = xml.xpath('//div[@class="group-misc"]')[0]
        intro = misc.xpath('string(.)') or ''
        if intro.find('退出小组') > -1:
            return 'joined'
        elif intro.find('你已经申请加入小组') > -1:
            return 'waiting'
        elif intro.find('申请加入小组') > -1:
            res = self.api.xml(API_GROUP_GROUP_HOME % group_alias, 'post', data={
                'ck': self.api.ck(),
                'action': 'request_join',
                'message': message,
                'send': '发送',
            })
            misc = res.xpath('//div[@class="group-misc"]')[0]
            intro = misc.xpath('string(.)') or ''
            if intro.find('你已经申请加入小组') > -1:
                return 'waiting'
            else:
                return 'initial'
        else:
            return 'initial'