def do_folder_create(client, args):
    """Create directory"""
    for folder_uri in args.uris:
        client.create_folder(folder_uri, recursive=True)
    return True