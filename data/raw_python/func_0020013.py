def create(self, comment, mentions=()):
        """
        create comment
        :param comment:
        :param mentions: list of pair of code and type("USER", "GROUP", and so on)
        :return:
        """

        data = {
            "app": self.app_id,
            "record": self.record_id,
            "comment": {
                "text": comment,
            }
        }

        if len(mentions) > 0:
            _mentions = []
            for m in mentions:
                if isinstance(m, (list, tuple)):
                    if len(m) == 2:
                        _mentions.append({
                            "code": m[0],
                            "type": m[1]
                        })
                    else:
                        raise Exception("mention have to have code and target type. ex.[('user_1', 'USER')]")
                elif isinstance(m, Mention):
                    _mentions.append(m.serialize())

            data["comment"]["mentions"] = _mentions

        resp = self._request("POST", self._url, data)
        r = cr.CreateCommentResult(resp)
        return r