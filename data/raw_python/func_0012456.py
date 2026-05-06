def parse_args(argv):
    """
    Use Argparse to parse command-line arguments.

    :param argv: list of arguments to parse (``sys.argv[1:]``)
    :type argv: ``list``
    :return: parsed arguments
    :rtype: :py:class:`argparse.Namespace`
    """
    p = argparse.ArgumentParser(
        description='pypi-download-stats - Calculate detailed download stats '
                    'and generate HTML and badges for PyPI packages - '
                    '<%s>' % PROJECT_URL,
        prog='pypi-download-stats'
    )
    p.add_argument('-V', '--version', action='version',
                   version='%(prog)s ' + VERSION)
    p.add_argument('-v', '--verbose', dest='verbose', action='count',
                   default=0,
                   help='verbose output. specify twice for debug-level output.')
    m = p.add_mutually_exclusive_group()
    m.add_argument('-Q', '--no-query', dest='query', action='store_false',
                   default=True, help='do not query; just generate output '
                                      'from cached data')
    m.add_argument('-G', '--no-generate', dest='generate', action='store_false',
                   default=True, help='do not generate output; just query '
                                      'data and cache results')
    p.add_argument('-o', '--out-dir', dest='out_dir', action='store', type=str,
                   default='./pypi-stats', help='output directory (default: '
                                                './pypi-stats')
    p.add_argument('-p', '--project-id', dest='project_id', action='store',
                   type=str, default=None,
                   help='ProjectID for your Google Cloud user, if not using '
                        'service account credentials JSON file')
    # @TODO this is tied to the DiskDataCache class
    p.add_argument('-c', '--cache-dir', dest='cache_dir', action='store',
                   type=str, default='./pypi-stats-cache',
                   help='stats cache directory (default: ./pypi-stats-cache)')
    p.add_argument('-B', '--backfill-num-days', dest='backfill_days', type=int,
                   action='store', default=7,
                   help='number of days of historical data to backfill, if '
                        'missing (defaut: 7). Note this may incur BigQuery '
                        'charges. Set to -1 to backfill all available history.')
    g = p.add_mutually_exclusive_group()
    g.add_argument('-P', '--project', dest='PROJECT', action='append', type=str,
                   help='project name to query/generate stats for (can be '
                        'specified more than once; '
                        'this will reduce query cost for multiple projects)')
    g.add_argument('-U', '--user', dest='user', action='store', type=str,
                   help='Run for all PyPI projects owned by the specified'
                        'user.')
    args = p.parse_args(argv)
    return args