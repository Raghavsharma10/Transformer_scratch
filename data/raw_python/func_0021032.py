def turn_off_syncing(for_post_save=True, for_post_delete=True, for_m2m_changed=True, for_post_bulk_operation=True):
    """
    Disables all of the signals for syncing entities. By default, everything is turned off. If the user wants
    to turn off everything but one signal, for example the post_save signal, they would do:

    turn_off_sync(for_post_save=False)
    """
    if for_post_save:
        post_save.disconnect(save_entity_signal_handler, dispatch_uid='save_entity_signal_handler')
    if for_post_delete:
        post_delete.disconnect(delete_entity_signal_handler, dispatch_uid='delete_entity_signal_handler')
    if for_m2m_changed:
        m2m_changed.disconnect(m2m_changed_entity_signal_handler, dispatch_uid='m2m_changed_entity_signal_handler')
    if for_post_bulk_operation:
        post_bulk_operation.disconnect(bulk_operation_signal_handler, dispatch_uid='bulk_operation_signal_handler')