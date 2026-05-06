def deconstruct(self):
        """
        Denormalize is always false migrations
        """
        name, path, args, kwargs = super(AssetsFileField, self).deconstruct()
        kwargs['denormalize'] = False
        return name, path, args, kwargs