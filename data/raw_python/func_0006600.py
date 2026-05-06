def service_provider(*services):
    """
    This is a class decorator that declares a class to provide a set of services.
    It is expected that the class has a no-arg constructor and will be instantiated
    as a singleton.
    """

    def real_decorator(clazz):
        instance = clazz()
        for service in services:
            global_lookup.add(service, instance)
        return clazz

    return real_decorator