def delete(self, new_parent=None, recursive=False):
        """
        Deletes a category. Optionally moves content to new category.
        Note: If category is in root, new_parent must be specified.

        :param int new_parent: (optional) Category ID of new parent
        :param bool recursive: recursively delete contents inside this category

        Example Usage::

        >>> import muddle
        >>> muddle.category(10).delete()
        """

        params = {'wsfunction': 'core_course_delete_categories',
                  'categories[0][id]': self.category_id,
                  'categories[0][recursive]': int(recursive)}
        if new_parent:
            params.update({'categories[0][newparent]': new_parent})
        params.update(self.request_params)

        return requests.post(self.api_url, params=params, verify=False)