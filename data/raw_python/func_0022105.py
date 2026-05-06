def service_pre_save(instance, *args, **kwargs):
    """
    Used to do a service full check when saving it.
    """

    # check if service is unique
    # we cannot use unique_together as it relies on a combination of fields
    # from different models (service, resource)
    exists = Service.objects.filter(url=instance.url,
                                    type=instance.type,
                                    catalog=instance.catalog).count() > 0

    # TODO: When saving from the django admin, this should not be triggered.
    # Reference: http://stackoverflow.com/questions/11561722/django-what-is-the-role-of-modelstate
    if instance._state.adding and exists:
        raise Exception("There is already such a service. url={0} catalog={1}".format(
            instance.url, instance.catalog
        ))