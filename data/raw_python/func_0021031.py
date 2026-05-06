def m2m_changed_entity_signal_handler(sender, instance, action, **kwargs):
    """
    Defines a signal handler for a manytomany changed signal. Only listens for the
    post actions so that entities are synced once (rather than twice for a pre and post action).
    """
    if action == 'post_add' or action == 'post_remove' or action == 'post_clear':
        save_entity_signal_handler(sender, instance, **kwargs)