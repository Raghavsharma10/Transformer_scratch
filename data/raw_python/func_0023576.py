def user_set_avatar(self, action=None, quick_key=None, url=None):
        """user/set_avatar

        http://www.mediafire.com/developers/core_api/1.3/user/#set_avatar
        """
        return self.request("user/set_avatar", QueryParams({
            "action": action,
            "quick_key": quick_key,
            "url": url
        }))