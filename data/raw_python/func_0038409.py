def operate_on(self, when=None, apply=False, **kwargs):
        """Do something with operate_on. If apply is True, all transactions will
        be applied and saved via celery task."""

        # get pzone based on id
        pzone = self.get(**kwargs)

        # cache the current time
        now = timezone.now()

        # ensure we have some value for when
        if when is None:
            when = now

        if when < now:
            histories = pzone.history.filter(date__lte=when)
            if histories.exists():
                # we have some history, use its data
                pzone.data = histories[0].data

        else:
            # only apply operations if cache is expired or empty, or we're looking at the future
            data = pzone.data

            # Get the cached time of the next expiration
            next_operation_time = cache.get('pzone-operation-expiry-' + pzone.name)
            if next_operation_time is None or next_operation_time < when:

                # start applying operations
                pending_operations = pzone.operations.filter(when__lte=when, applied=False)
                for operation in pending_operations:
                    data = operation.apply(data)

                # reassign data
                pzone.data = data

                if apply and pending_operations.exists():
                    # there are operations to apply, do celery task
                    update_pzone.delay(**kwargs)

        # return pzone, modified if apply was True
        return pzone