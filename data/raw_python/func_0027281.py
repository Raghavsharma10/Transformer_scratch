def delete_error_message(sender, instance, name, source, target, **kwargs):
    """ Delete error message if instance state changed from erred """
    if source != StateMixin.States.ERRED:
        return
    instance.error_message = ''
    instance.save(update_fields=['error_message'])