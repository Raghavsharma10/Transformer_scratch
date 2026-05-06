def handle_resource_not_found(resource):
    """
    Set resource state to ERRED and append/create "not found" error message.
    """
    resource.set_erred()
    resource.runtime_state = ''
    message = 'Does not exist at backend.'
    if message not in resource.error_message:
        if not resource.error_message:
            resource.error_message = message
        else:
            resource.error_message += ' (%s)' % message
    resource.save()
    logger.warning('%s %s (PK: %s) does not exist at backend.' % (
        resource.__class__.__name__, resource, resource.pk))