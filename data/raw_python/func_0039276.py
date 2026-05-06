def sync_local_to_changes(org, syncer, fetches, deleted_fetches, progress_callback=None):
    """
    Sync local instances against iterators which return fetches of changed and deleted remote objects.

    :param * org: the org
    :param * syncer: the local model syncer
    :param * fetches: an iterator returning fetches of modified remote objects
    :param * deleted_fetches: an iterator returning fetches of deleted remote objects
    :param * progress_callback: callable for tracking progress - called for each fetch with number of contacts fetched
    :return: tuple containing counts of created, updated and deleted local instances
    """
    num_synced = 0
    outcome_counts = defaultdict(int)

    for fetch in fetches:
        for remote in fetch:
            outcome = sync_from_remote(org, syncer, remote)
            outcome_counts[outcome] += 1

        num_synced += len(fetch)
        if progress_callback:
            progress_callback(num_synced)

    # any item that has been deleted remotely should also be released locally
    for deleted_fetch in deleted_fetches:
        for deleted_remote in deleted_fetch:
            identity = syncer.identify_remote(deleted_remote)
            with syncer.lock(org, identity):
                existing = syncer.fetch_local(org, identity)
                if existing:
                    syncer.delete_local(existing)
                    outcome_counts[SyncOutcome.deleted] += 1

        num_synced += len(deleted_fetch)
        if progress_callback:
            progress_callback(num_synced)

    return (
        outcome_counts[SyncOutcome.created],
        outcome_counts[SyncOutcome.updated],
        outcome_counts[SyncOutcome.deleted],
        outcome_counts[SyncOutcome.ignored],
    )