def delete_service_settings_on_scope_delete(sender, instance, **kwargs):
    """ If VM that contains service settings were deleted - all settings
        resources could be safely deleted from NC.
    """
    for service_settings in ServiceSettings.objects.filter(scope=instance):
        service_settings.unlink_descendants()
        service_settings.delete()