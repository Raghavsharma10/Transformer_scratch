def remove(self, membership_id):
        """Remove a member from the group.

        :param str membership_id: the ID of a member in this group
        :return: ``True`` if the member was successfully removed
        :rtype: bool
        """
        path = '{}/remove'.format(membership_id)
        url = utils.urljoin(self.url, path)
        payload = {'membership_id': membership_id}
        response = self.session.post(url, json=payload)
        return response.ok