def count_quota_handler_factory(count_quota_field):
    """ Creates handler that will recalculate count_quota on creation/deletion """

    def recalculate_count_quota(sender, instance, **kwargs):
        signal = kwargs['signal']
        if signal == signals.post_save and kwargs.get('created'):
            count_quota_field.add_usage(instance, delta=1)
        elif signal == signals.post_delete:
            count_quota_field.add_usage(instance, delta=-1, fail_silently=True)

    return recalculate_count_quota