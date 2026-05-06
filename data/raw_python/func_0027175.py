def delete_service_settings_on_service_delete(sender, instance, **kwargs):
    """ Delete not shared service settings without services """
    service = instance
    try:
        service_settings = service.settings
    except ServiceSettings.DoesNotExist:
        # If this handler works together with delete_service_settings_on_scope_delete
        # it tries to delete service settings that are already deleted.
        return
    if not service_settings.shared:
        service.settings.delete()