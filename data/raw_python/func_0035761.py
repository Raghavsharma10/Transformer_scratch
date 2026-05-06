def list_dataset_uris(cls, base_uri, config_path):
        """Return list containing URIs in location given by base_uri."""

        parsed_uri = generous_parse_uri(base_uri)
        uri_list = []

        path = parsed_uri.path
        if IS_WINDOWS:
            path = unix_to_windows_path(parsed_uri.path, parsed_uri.netloc)

        for d in os.listdir(path):
            dir_path = os.path.join(path, d)

            if not os.path.isdir(dir_path):
                continue

            storage_broker = cls(dir_path, config_path)

            if not storage_broker.has_admin_metadata():
                continue

            uri = storage_broker.generate_uri(
                name=d,
                uuid=None,
                base_uri=base_uri
            )
            uri_list.append(uri)

        return uri_list