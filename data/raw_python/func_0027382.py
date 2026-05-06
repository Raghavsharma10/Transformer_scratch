def revoke_auth_access(self, access_token):
        """
        授权回收接口，帮助开发者主动取消用户的授权。

        应用下线时，清空所有用户的授权
        应用新上线了功能，需要取得用户scope权限，可以回收后重新引导用户授权
        开发者调试应用，需要反复调试授权功能
        应用内实现类似登出微博帐号的功能

        并传递给你以下参数，source：应用appkey，uid ：取消授权的用户，auth_end ：取消授权的时间

        :param access_token:
        :return: bool
        """
        result = self.request("post", "revokeoauth2", data={"access_token": access_token})
        return bool(result.get("result"))