def turn_on_syncing(for_post_save=True, for_post_delete=True, for_m2m_changed=True, for_post_bulk_operation=False):
    """
    Enables all of the signals for syncing entities. Everything is True by default, except for the post_bulk_operation
    signal. The reason for this is because when any bulk operation occurs on any mirrored entity model, it will
    result in every single entity being synced again. This is not a desired behavior by the majority of users, and
    should only be turned on explicitly.
    """
    if for_post_save:
        post_save.connect(save_entity_signal_handler, dispatch_uid='save_entity_signal_handler')
    if for_post_delete:
        post_delete.connect(delete_entity_signal_handler, dispatch_uid='delete_entity_signal_handler')
    if for_m2m_changed:
        m2m_changed.connect(m2m_changed_entity_signal_handler, dispatch_uid='m2m_changed_entity_signal_handler')
    if for_post_bulk_operation:
        post_bulk_operation.connect(bulk_operation_signal_handler, dispatch_uid='bulk_operation_signal_handler')