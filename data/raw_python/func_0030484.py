def bundle_new(args, l, rc):
    """Create a new bundle"""

    from ambry.orm.exc import ConflictError

    d = dict(
        dataset=args.dataset,
        revision=args.revision,
        source=args.source,
        bspace=args.space,
        subset=args.subset,
        btime=args.time,
        variation=args.variation)

    try:
        ambry_account = rc.accounts.get('ambry', {})
    except:
        ambry_account = None

    if not ambry_account:
        fatal("Failed to get an accounts.ambry entry from the configuration. ")

    if not ambry_account.get('name') or not ambry_account.get('email'):
        fatal('Must set accounts.ambry.email and accounts.ambry.name n account config file')

    if args.dryrun:
        from ..identity import Identity
        d['revision'] = 1
        d['id'] = 'dXXX'
        print(str(Identity.from_dict(d)))
        return

    try:
        b = l.new_bundle(assignment_class=args.key, **d)

        if ambry_account:
            b.metadata.contacts.wrangler = ambry_account

        b.build_source_files.bundle_meta.objects_to_record()
        b.commit()

    except ConflictError:
        fatal("Can't create dataset; one with a conflicting name already exists")

    print(b.identity.fqname)