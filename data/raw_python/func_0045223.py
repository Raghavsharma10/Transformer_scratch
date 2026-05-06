def delete_tag(self, tag_id):
        """
        Deletes a Tag to current object

        :param tag_id: the id of the tag which should be deleted
        :type tag_id: int
        :rtype: None
        """
        from highton.models.tag import Tag

        self._delete_request(
            endpoint=self.ENDPOINT + '/' + str(self.id) + '/' + Tag.ENDPOINT + '/' + str(tag_id),
        )