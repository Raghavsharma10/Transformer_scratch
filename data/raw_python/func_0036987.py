def command(state, args):
    """Purge all caches."""
    state.cache_manager.teardown()
    state.cache_manager.setup()
    EpisodeTypes.forget(state.db)
    del state.file_picker