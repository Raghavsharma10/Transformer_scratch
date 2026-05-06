def reset_crops(self):
        """
        Reset all known crops to the default crop.

        If settings.ASSET_CELERY is specified then
        the task will be run async
        """

        if self._can_crop():
            if settings.CELERY or settings.USE_CELERY_DECORATOR:
                # this means that we are using celery
                tasks.reset_crops.apply_async(args=[self.pk], countdown=5)
            else:
                tasks.reset_crops(None, asset=self)