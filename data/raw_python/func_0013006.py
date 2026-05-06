def process_request(self, unused_request):
    """Called by Django before deciding which view to execute."""
    # Compare to the first half of toplevel() in context.py.
    tasklets._state.clear_all_pending()
    # Create and install a new context.
    ctx = tasklets.make_default_context()
    tasklets.set_context(ctx)