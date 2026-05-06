def sync_from_remote(org, syncer, remote):
    """
    Sync local instance against a single remote object

    :param * org: the org
    :param * syncer: the local model syncer
    :param * remote: the remote object
    :return: the outcome (created, updated or deleted)
    """
    identity = syncer.identify_remote(remote)

    with syncer.lock(org, identity):
        existing = syncer.fetch_local(org, identity)

        # derive kwargs for the local model (none return here means don't keep)
        remote_as_kwargs = syncer.local_kwargs(org, remote)

        # exists locally
        if existing:
            existing.org = org  # saves pre-fetching since we already have the org

            if remote_as_kwargs:
                if syncer.update_required(existing, remote, remote_as_kwargs) or not existing.is_active:
                    for field, value in remote_as_kwargs.items():
                        setattr(existing, field, value)

                    existing.is_active = True
                    existing.save()
                    return SyncOutcome.updated

            elif existing.is_active:  # exists locally, but shouldn't now to due to model changes
                syncer.delete_local(existing)
                return SyncOutcome.deleted

        elif remote_as_kwargs:
            syncer.model.objects.create(**remote_as_kwargs)
            return SyncOutcome.created

    return SyncOutcome.ignored