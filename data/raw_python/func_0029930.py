def root_sync(args, l, config):
    """Sync with the remote. For more options, use library sync
    """
    from requests.exceptions import ConnectionError

    all_remote_names = [ r.short_name for r in l.remotes ]

    if args.all:
        remotes = all_remote_names
    else:
        remotes = args.refs

    prt("Sync with {} remotes or bundles ".format(len(remotes)))

    if not remotes:
        return

    for ref in remotes:
        l.commit()

        try:
            if ref in all_remote_names: # It's a remote name
                l.sync_remote(l.remote(ref))

            else: # It's a bundle reference

                l.checkin_remote_bundle(ref)

        except NotFoundError as e:
            warn(e)
            continue
        except ConnectionError as e:
            warn(e)
            continue