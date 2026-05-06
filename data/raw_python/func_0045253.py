def list_comments(self, page=0):
        """
        Get the comments of current object

        :param page: the page starting at 0
        :return: the emails
        :rtype: list
        """
        from highton.models.comment import Comment
        params = {'page': int(page) * self.COMMENT_OFFSET}

        return fields.ListField(
            name=self.ENDPOINT,
            init_class=Comment
        ).decode(
            self.element_from_string(
                self._get_request(
                    endpoint=self.ENDPOINT + '/' + str(self.id) + '/' + Comment.ENDPOINT,
                    params=params
                ).text
            )
        )