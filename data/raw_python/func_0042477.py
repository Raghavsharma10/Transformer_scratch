def publish(self, user=None, when=None):
        """
        Publishes a item and any sub items.
        A new transaction will be started if
        we aren't already in a transaction.

        Should only be run on draft items
        """

        assert self.state == self.DRAFT

        user_published = 'code'
        if user:
            user_published = user.username

        now = timezone.now()

        with xact():
            # If this item hasn't got live yet and no new date was specified
            # delete the old scheduled items and schedule this one on that date
            published = False
            if getattr(self._meta, '_is_view', False):
                published = self.is_published
            else:
                published = self.object.is_published

            if not when and not published and self.last_scheduled:
                klass = self.get_version_class()
                for obj in klass.normal.filter(object_id=self.object_id,
                                               last_scheduled=self.last_scheduled,
                                               state=self.SCHEDULED):
                    when = self.date_published
                    obj.delete()

            when = when or now

            # Drafts get preserved so save the
            # time we last cloned this
            if self.state == self.DRAFT:
                self.last_scheduled = now
                self.date_published = when
                self.save(last_save=now)

            self._clone()

            self.user_published = user_published
            self.state = self.SCHEDULED
            self.save()

            self.schedule(when=when)