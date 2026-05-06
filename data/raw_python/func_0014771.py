def delete(self, *args, **kwargs):
        """
        This method implements retries for object deletion.
        """
        count = 0
        max_retries=3
        while True:
            try:
                return super(BaseModel, self).delete(*args, **kwargs)
            except django.db.utils.OperationalError:
                if count >= max_retries:
                    raise
                count += 1