def do_pending_lookups(event, sender, **kwargs):
    """Handle any pending relations to the sending model.
Sent from class_prepared."""
    key = (sender._meta.app_label, sender._meta.name)
    for callback in pending_lookups.pop(key, []):
        callback(sender)