def leave_group(self, group_alias):
        """
        退出小组
        
        :param group_alias: 小组ID
        :return: 
        """
        return self.api.req(API_GROUP_GROUP_HOME % group_alias, params={
            'action': 'quit',
            'ck': self.api.ck(),
        })