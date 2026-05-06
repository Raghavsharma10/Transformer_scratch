def get_replies(self):
        """Get the replies to this comment."""
        url = self._imgur._base_url + "/3/comment/{0}/replies".format(self.id)
        json = self._imgur._send_request(url)
        child_comments = json['children']
        return [Comment(com, self._imgur) for com in child_comments]