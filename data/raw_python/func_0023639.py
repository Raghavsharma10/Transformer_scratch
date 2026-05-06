def do_file_download(client, args):
    """Download file"""

    # Sanity check
    if not os.path.isdir(args.dest_path) and not args.dest_path.endswith('/'):
        print("file-download: "
              "target '{}' is not a directory".format(args.dest_path))
        if not os.path.exists(args.dest_path):
            print("\tHint: add trailing / to create one")
        return None

    for src_uri in args.uris:
        print("Downloading {} to {}".format(src_uri, args.dest_path))
        client.download_file(src_uri, args.dest_path)
        print("Downloaded {}".format(src_uri))

    return True