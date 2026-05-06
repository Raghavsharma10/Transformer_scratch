def handle_resource_update_success(resource):
    """
    Recover resource if its state is ERRED and clear error message.
    """
    update_fields = []
    if resource.state == resource.States.ERRED:
        resource.recover()
        update_fields.append('state')

    if resource.state in (resource.States.UPDATING, resource.States.CREATING):
        resource.set_ok()
        update_fields.append('state')

    if resource.error_message:
        resource.error_message = ''
        update_fields.append('error_message')

    if update_fields:
        resource.save(update_fields=update_fields)
    logger.warning('%s %s (PK: %s) was successfully updated.' % (
        resource.__class__.__name__, resource, resource.pk))