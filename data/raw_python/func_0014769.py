def save(self, *args, **kwargs):
        """
        This save method protects against two processesses concurrently modifying
        the same object. Normally the second save would silently overwrite the
        changes from the first. Instead we raise a ConcurrentModificationError.
        """
        cls = self.__class__
        if self.pk:
            rows = cls.objects.filter(
                pk=self.pk, _change=self._change).update(
                _change=self._change + 1)
            if not rows:
                raise ConcurrentModificationError(cls.__name__, self.pk)
            self._change += 1

        count = 0
        max_retries=3
        while True:
            try:
                return super(BaseModel, self).save(*args, **kwargs)
            except django.db.utils.OperationalError:
                if count >= max_retries:
                    raise
                count += 1