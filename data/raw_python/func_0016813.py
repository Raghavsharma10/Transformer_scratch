def dl_cub(cub_url, cub_archive_name):
    """ Download cub archive from cub_url and store it in cub_archive_name """
    with open(cub_archive_name, 'wb') as f:
        remote_file = urllib2.urlopen(cub_url)
        meta = remote_file.info()

        # The server may provide us with the size of the file.
        cl_header = meta.getheaders("Content-Length")
        remote_file_size = int(cl_header[0]) if len(cl_header) > 0 else None

        # Initialise variables
        local_file_size = 0
        block_size = 128*1024

        # Do the download
        while True:
            data = remote_file.read(block_size)

            if not data:
                break

            f.write(data)
            local_file_size += len(data)

        if (remote_file_size is not None and
                not local_file_size == remote_file_size):
            log.warn("Local file size '{}' "
                "does not match remote '{}'".format(
                    local_file_size, remote_file_size))

        remote_file.close()