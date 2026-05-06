def do_file_update_metadata(client, args):
    """Update file metadata"""
    client.update_file_metadata(args.uri, filename=args.filename,
                                description=args.description, mtime=args.mtime,
                                privacy=args.privacy)
    return True