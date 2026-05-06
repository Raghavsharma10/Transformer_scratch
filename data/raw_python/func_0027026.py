def get_permitted_objects_uuids(cls, user):
        """
        Return query dictionary to search objects available to user.
        """
        uuids = filter_queryset_for_user(cls.objects.all(), user).values_list('uuid', flat=True)
        key = core_utils.camel_case_to_underscore(cls.__name__) + '_uuid'
        return {key: uuids}