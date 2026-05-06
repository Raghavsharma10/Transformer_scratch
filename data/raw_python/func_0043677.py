def new(cls, settings, *args, **kwargs):
        """
        Create a new Cloud instance based on the Settings
        """
        logger.debug('Initializing new "%s" Instance object' % settings['CLOUD'])
        cloud = settings['CLOUD']
        if cloud == 'bare':
            self = BareInstance(settings=settings, *args, **kwargs)
        elif cloud == 'aws':
            self = AWSInstance(settings=settings, *args, **kwargs)
        elif cloud == 'gcp':
            self = GCPInstance(settings=settings, *args, **kwargs)
        else:
            raise DSBException('Cloud "%s" not supported' % cloud)
        return self