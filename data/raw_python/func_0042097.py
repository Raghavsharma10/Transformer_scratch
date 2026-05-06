def ensure_crops(self, *required_crops):
        """
        Make sure a crop exists for each crop in required_crops.
        Existing crops will not be changed.

        If settings.ASSET_CELERY is specified then
        the task will be run async
        """
        if self._can_crop():
            if settings.CELERY or settings.USE_CELERY_DECORATOR:
                # this means that we are using celery
                args = [self.pk]+list(required_crops)
                tasks.ensure_crops.apply_async(args=args, countdown=5)
            else:
                tasks.ensure_crops(None, *required_crops, asset=self)