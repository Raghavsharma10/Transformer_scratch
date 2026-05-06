def do_resource_delete(client, args):
    """Remove resource"""
    for resource_uri in args.uris:
        client.delete_resource(resource_uri, purge=args.purge)
        print("Deleted {}".format(resource_uri))
    return True