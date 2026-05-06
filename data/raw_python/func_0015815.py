def get_json(self, link):
        """
        Returns specified link instance as JSON.

        :param link: the link instance.
        :rtype: JSON.
        """
        return json.dumps({
            'id':           link.id,
            'title':        link.title,
            'url':          link.get_absolute_url(),
            'edit_link':    reverse(
                '{0}:edit'.format(self.url_namespace),
                kwargs = {'pk': link.pk}
            ),
        })