def update_pulled_fields(instance, imported_instance, fields):
    """
    Update instance fields based on imported from backend data.
    Save changes to DB only one or more fields were changed.
    """
    modified = False
    for field in fields:
        pulled_value = getattr(imported_instance, field)
        current_value = getattr(instance, field)
        if current_value != pulled_value:
            setattr(instance, field, pulled_value)
            logger.info("%s's with PK %s %s field updated from value '%s' to value '%s'",
                        instance.__class__.__name__, instance.pk, field, current_value, pulled_value)
            modified = True
    error_message = getattr(imported_instance, 'error_message', '') or getattr(instance, 'error_message', '')
    if error_message and instance.error_message != error_message:
        instance.error_message = imported_instance.error_message
        modified = True
    if modified:
        instance.save()