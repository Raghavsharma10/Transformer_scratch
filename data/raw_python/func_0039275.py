def sync_local_to_set(org, syncer, remote_set):
    """
    Syncs an org's set of local instances of a model to match the set of remote objects. Local objects not in the remote
    set are deleted.

    :param org: the org
    :param * syncer: the local model syncer
    :param remote_set: the set of remote objects
    :return: tuple of number of local objects created, updated, deleted and ignored
    """
    outcome_counts = defaultdict(int)

    remote_identities = set()

    for remote in remote_set:
        outcome = sync_from_remote(org, syncer, remote)
        outcome_counts[outcome] += 1

        remote_identities.add(syncer.identify_remote(remote))

    # active local objects which weren't in the remote set need to be deleted
    active_locals = syncer.fetch_all(org).filter(is_active=True)
    delete_locals = active_locals.exclude(**{syncer.local_id_attr + "__in": remote_identities})

    for local in delete_locals:
        with syncer.lock(org, syncer.identify_local(local)):
            syncer.delete_local(local)
            outcome_counts[SyncOutcome.deleted] += 1

    return (
        outcome_counts[SyncOutcome.created],
        outcome_counts[SyncOutcome.updated],
        outcome_counts[SyncOutcome.deleted],
        outcome_counts[SyncOutcome.ignored],
    )