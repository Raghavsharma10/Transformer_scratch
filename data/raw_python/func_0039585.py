def post_process(self, paths, dry_run=False, **options):
        """
        Overridden to work around https://code.djangoproject.com/ticket/19111
        """
        with post_process_error_counter(self):
            with patched_name_fn(self, 'hashed_name', 'hashed name'):
                with patched_name_fn(self, 'url', 'url'):
                    for result in super(LaxPostProcessorMixin,
                                        self).post_process(paths, dry_run, **options):
                        yield result
            error_count = self._post_process_error_count
            if error_count:
                print('%s post-processing error%s.' % (error_count,
                        '' if error_count == 1 else 's'))