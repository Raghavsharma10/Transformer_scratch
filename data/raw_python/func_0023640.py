def do_file_show(client, args):
    """Output file contents to stdout"""
    for src_uri in args.uris:
        client.download_file(src_uri, sys.stdout.buffer)

    return True