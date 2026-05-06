def extractUserStore(userAccount, extractionDestination, legacySiteAuthoritative=True):
    """
    Move the SubStore for the given user account out of the given site store
    completely.  Place the user store's database directory into the given
    destination directory.

    @type userAccount: C{LoginAccount}
    @type extractionDestination: C{FilePath}

    @type legacySiteAuthoritative: C{bool}

    @param legacySiteAuthoritative: before moving the user store, clear its
    authentication information, copy that which is associated with it in the
    site store rather than trusting its own.  Currently this flag is necessary
    (and defaults to true) because things like the ClickChronicle
    password-changer gizmo still operate on the site store.

    """
    if legacySiteAuthoritative:
        # migrateDown() manages its own transactions, since it is copying items
        # between two different stores.
        userAccount.migrateDown()
    av = userAccount.avatars
    av.open().close()
    def _():
        # We're separately deleting several Items from the site store, then
        # we're moving some files.  If we cannot move the files, we don't want
        # to delete the items.

        # There is one unaccounted failure mode here: if the destination of the
        # move is on a different mount point, the moveTo operation will fall
        # back to a non-atomic copy; if all of the copying succeeds, but then
        # part of the deletion of the source files fails, we will be left
        # without a complete store in this site store's files directory, but
        # the account Items will remain.  This will cause odd errors on login
        # and at other unpredictable times.  The database is only one file, so
        # we will either remove it all or none of it.  Resolving this requires
        # manual intervention currently: delete the substore's database
        # directory and the account items (LoginAccount and LoginMethods)
        # manually.

        # However, this failure is extremely unlikely, as it would almost
        # certainly indicate a misconfiguration of the permissions on the site
        # store's files area.  As described above, a failure of the call to
        # os.rename(), if the platform's rename is atomic (which it generally
        # is assumed to be) will not move any files and will cause a revert of
        # the transaction which would have deleted the accompanying items.

        av.deleteFromStore()
        userAccount.deleteLoginMethods()
        userAccount.deleteFromStore()
        av.storepath.moveTo(extractionDestination)
    userAccount.store.transact(_)