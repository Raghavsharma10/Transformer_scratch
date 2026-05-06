def invite(name, emails, perm):
    """Invite other users to access the virtual folder.

    \b
    NAME: Name of a virtual folder.
    EMAIL: Emails to invite.
    """
    with Session() as session:
        try:
            assert perm in ['rw', 'ro'], \
                   'Invalid permission: {}'.format(perm)
            result = session.VFolder(name).invite(perm, emails)
            invited_ids = result.get('invited_ids', [])
            if len(invited_ids) > 0:
                print('Invitation sent to:')
                for invitee in invited_ids:
                    print('\t- ' + invitee)
            else:
                print('No users found. Invitation was not sent.')
        except Exception as e:
            print_error(e)
            sys.exit(1)