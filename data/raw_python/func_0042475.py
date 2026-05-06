def get_version(self, state=None, date=None):
        """
        Get a particular version of an item

        :param state: The state you want to get.
        :param date: Get a version that was published before or on this date.
        """

        version_model = self._meta._version_model
        q = version_model.objects.filter(object_id=self.pk)
        if state:
            q = version_model.normal.filter(object_id=self.pk, state=state)

        if date:
            q = q.filter(date_published__lte=date)

        q = q.order_by('-date_published')

        results = q[:1]
        if results:
            return results[0]
        return None