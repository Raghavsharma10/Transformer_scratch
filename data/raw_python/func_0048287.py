def register(klass):
    "Register a class into the agnocomplete registry."
    logger.info("registering {}".format(klass.__name__))
    AGNOCOMPLETE_REGISTRY[klass.slug] = klass