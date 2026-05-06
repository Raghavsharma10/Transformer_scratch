def main():  # pylint: disable=too-many-statements
    """Main entry point"""
    parser = argparse.ArgumentParser(prog='mediafire-cli',
                                     description=__doc__)

    parser.add_argument('--debug', dest='debug', action='store_true',
                        default=False, help='Enable debug output')
    parser.add_argument('--email', dest='email', required=False,
                        default=os.environ.get('MEDIAFIRE_EMAIL', None))
    parser.add_argument('--password', dest='password', required=False,
                        default=os.environ.get('MEDIAFIRE_PASSWORD', None))

    actions = parser.add_subparsers(title='Actions', dest='action')
    # http://bugs.python.org/issue9253#msg186387
    actions.required = True

    # ls
    subparser = actions.add_parser('ls',
                                   help=do_ls.__doc__)
    subparser.add_argument('uri', nargs='?',
                           help='MediaFire URI',
                           default='mf:///')

    # file-upload
    subparser = actions.add_parser('file-upload',
                                   help=do_file_upload.__doc__)
    subparser.add_argument('paths', nargs='+',
                           help='Path[s] to upload')
    subparser.add_argument('dest_uri', help='Destination MediaFire URI')

    # file-download
    subparser = actions.add_parser('file-download',
                                   help=do_file_download.__doc__)
    subparser.add_argument('uris', nargs='+',
                           help='MediaFire File URI[s] to download')
    subparser.add_argument('dest_path', help='Destination path')

    # file-show
    subparser = actions.add_parser('file-show',
                                   help=do_file_show.__doc__)
    subparser.add_argument('uris', nargs='+',
                           help='MediaFire File URI[s] to print out')

    # folder-create
    subparser = actions.add_parser('folder-create',
                                   help=do_folder_create.__doc__)
    subparser.add_argument('uris', nargs='+',
                           help='MediaFire folder path URI[s]')

    # resource-delete
    subparser = actions.add_parser('resource-delete',
                                   help=do_resource_delete.__doc__)
    subparser.add_argument('uris', nargs='+',
                           help='MediaFire resource URI[s]')
    subparser.add_argument('--purge', help="Purge, don't send to trash",
                           dest="purge", action="store_true", default=False)

    # file-update-metadata
    subparser = actions.add_parser('file-update-metadata',
                                   help=do_file_update_metadata.__doc__)
    subparser.add_argument('uri', help='MediaFire file URI')
    subparser.add_argument('--filename', help='Set file name',
                           default=None, dest='filename')
    subparser.add_argument('--privacy', help='Set file privacy',
                           choices=['public', 'private'],
                           default=None, dest='privacy')
    subparser.add_argument('--description',
                           help='Set file description',
                           dest='description', default=None)
    subparser.add_argument('--mtime', help="Set file modification time",
                           dest='mtime', default=None)

    # folder-update-metadata
    subparser = actions.add_parser('folder-update-metadata',
                                   help=do_folder_update_metadata.__doc__)
    subparser.add_argument('uri', help='MediaFire folder URI')
    subparser.add_argument('--foldername', help='Set folder name',
                           default=None, dest='foldername')
    subparser.add_argument('--privacy', help='Set folder privacy',
                           choices=['public', 'private'],
                           default=None, dest='privacy')
    subparser.add_argument('--recursive', help='Set privacy recursively',
                           action='store_true', default=None,
                           dest='recursive')
    subparser.add_argument('--description',
                           help='Set folder description',
                           dest='description', default=None)
    subparser.add_argument('--mtime', help='Set folder mtime',
                           default=None, dest='mtime')

    # debug-get-resource
    subparser = actions.add_parser('debug-get-resource',
                                   help=do_debug_get_resource.__doc__)
    subparser.add_argument('uri', help='MediaFire resource URI',
                           default='mediafire:/', nargs='?')

    args = parser.parse_args()

    if args.debug:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        logging.getLogger("mediafire.client").setLevel(logging.DEBUG)

    client = MediaFireClient()

    if args.email and args.password:
        client.login(args.email, args.password, app_id=APP_ID)

    router = {
        "file-upload": do_file_upload,
        "file-download": do_file_download,
        "file-show": do_file_show,
        "ls": do_ls,
        "folder-create": do_folder_create,
        "resource-delete": do_resource_delete,
        "file-update-metadata": do_file_update_metadata,
        "folder-update-metadata": do_folder_update_metadata,
        "debug-get-resource": do_debug_get_resource
    }

    if args.action in router:
        result = router[args.action](client, args)

        if not result:
            sys.exit(1)
    else:
        print('Unsupported action: {}'.format(args.action))
        sys.exit(1)