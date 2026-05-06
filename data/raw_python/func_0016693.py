def large_size(self, as_string=True):
        """Returns a thumbnail's large size."""
        size = getattr(settings, 'USER_MEDIA_THUMB_SIZE_LARGE', (150, 150))
        if as_string:
            return u'{}x{}'.format(size[0], size[1])
        return size