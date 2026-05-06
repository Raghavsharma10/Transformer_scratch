def main(config, log):
    """Main function. Runs the program.

    :param dict config: Dictionary from get_arguments().
    :param logging.Logger log: Logger for this function. Populated by with_log() decorator.
    """
    validate(config)
    paths_and_urls = get_urls(config)
    if not paths_and_urls:
        log.warning('No artifacts; nothing to download.')
        return

    # Download files.
    total_size = 0
    chunk_size = max(min(max(v[1] for v in paths_and_urls.values()) // 50, 1048576), 1024)
    log.info('Downloading file%s (1 dot ~ %d KiB):', '' if len(paths_and_urls) == 1 else 's', chunk_size // 1024)
    for size, local_path, url in sorted((v[1], k, v[0]) for k, v in paths_and_urls.items()):
        download_file(config, local_path, url, size, chunk_size)
        total_size += size
        if config['mangle_coverage']:
            mangle_coverage(local_path)

    log.info('Downloaded %d file(s), %d bytes total.', len(paths_and_urls), total_size)