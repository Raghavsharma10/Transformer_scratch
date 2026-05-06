def list_tags(self):
        """
        Get the tags of current object

        :return: the tags
        :rtype: list
        """
        from highton.models.tag import Tag
        return fields.ListField(
            name=self.ENDPOINT,
            init_class=Tag
        ).decode(
            self.element_from_string(
                self._get_request(
                    endpoint=self.ENDPOINT + '/' + str(self.id) + '/' + Tag.ENDPOINT,
                ).text
            )
        )