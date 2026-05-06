def query_build_version(config, log):
    """Find the build version we're looking for.

    AppVeyor calls build IDs "versions" which is confusing but whatever. Job IDs aren't available in the history query,
    only on latest, specific version, and deployment queries. Hence we need two queries to get a one-time status update.

    Returns None if the job isn't queued yet.

    :raise HandledError: On invalid JSON data.

    :param dict config: Dictionary from get_arguments().
    :param logging.Logger log: Logger for this function. Populated by with_log() decorator.

    :return: Build version.
    :rtype: str
    """
    url = '/projects/{0}/{1}/history?recordsNumber=10'.format(config['owner'], config['repo'])

    # Query history.
    log.debug('Querying AppVeyor history API for %s/%s...', config['owner'], config['repo'])
    json_data = query_api(url)
    if 'builds' not in json_data:
        log.error('Bad JSON reply: "builds" key missing.')
        raise HandledError

    # Find AppVeyor build "version".
    for build in json_data['builds']:
        if config['tag'] and config['tag'] == build.get('tag'):
            log.debug('This is a tag build.')
        elif config['pull_request'] and config['pull_request'] == build.get('pullRequestId'):
            log.debug('This is a pull request build.')
        elif config['commit'] == build['commitId']:
            log.debug('This is a branch build.')
        else:
            continue
        log.debug('Build JSON dict: %s', str(build))
        return build['version']
    return None