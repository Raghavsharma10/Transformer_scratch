def invitations():
    """List and manage received invitations.
    """
    with Session() as session:
        try:
            result = session.VFolder.invitations()
            invitations = result.get('invitations', [])
            if len(invitations) < 1:
                print('No invitations.')
                return
            print('List of invitations (inviter, vfolder id, permission):')
            for cnt, inv in enumerate(invitations):
                if inv['perm'] == 'rw':
                    perm = 'read-write'
                elif inv['perm'] == 'ro':
                    perm = 'read-only'
                else:
                    perm = inv['perm']
                print('[{}] {}, {}, {}'.format(cnt + 1, inv['inviter'],
                                               inv['vfolder_id'], perm))

            selection = input('Choose invitation number to manage: ')
            if selection.isdigit():
                selection = int(selection) - 1
            else:
                return
            if 0 <= selection < len(invitations):
                while True:
                    action = input('Choose action. (a)ccept, (r)eject, (c)ancel: ')
                    if action.lower() == 'a':
                        # TODO: Let user can select access_key among many.
                        #       Currently, the config objects holds only one key.
                        config = get_config()
                        result = session.VFolder.accept_invitation(
                            invitations[selection]['id'], config.access_key)
                        print(result['msg'])
                        break
                    elif action.lower() == 'r':
                        result = session.VFolder.delete_invitation(
                            invitations[selection]['id'])
                        print(result['msg'])
                        break
                    elif action.lower() == 'c':
                        break
        except Exception as e:
            print_error(e)
            sys.exit(1)