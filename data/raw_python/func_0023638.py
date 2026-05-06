def do_file_upload(client, args):
    """Upload files"""

    # Sanity check
    if len(args.paths) > 1:
        # destination must be a directory
        try:
            resource = client.get_resource_by_uri(args.dest_uri)
        except ResourceNotFoundError:
            resource = None

        if resource and not isinstance(resource, Folder):
            print("file-upload: "
                  "target '{}' is not a directory".format(args.dest_uri))
            return None

    with client.upload_session():
        for src_path in args.paths:
            print("Uploading {} to {}".format(src_path, args.dest_uri))
            result = client.upload_file(src_path, args.dest_uri)

            print("Uploaded {}, result={}".format(src_path, result))

    return True