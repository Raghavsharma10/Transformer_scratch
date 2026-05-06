def load_prefix(s3_loc, success_only=None, recent_versions=None, exclude_regex=None, just_sql=False):
    """Get a bash command which will load every dataset in a bucket at a prefix.

    For this to work, all datasets must be of the form `s3://$BUCKET_NAME/$PREFIX/$DATASET_NAME/v$VERSION/$PARTITIONS`.
    Any other formats will be ignored.

    :param bucket_name
    :param prefix
    """
    bucket_name, prefix = _get_bucket_and_prefix(s3_loc)
    datasets = _get_common_prefixes(bucket_name, prefix)
    bash_cmd = ''

    for dataset in datasets:
        dataset = _remove_trailing_backslash(dataset)
        try:
            bash_cmd += get_bash_cmd('s3://{}/{}'.format(bucket_name, dataset),
                                     success_only=success_only, recent_versions=recent_versions,
                                     exclude_regex=exclude_regex, just_sql=just_sql)
        except Exception as e:
            sys.stderr.write('Failed to process {}, {}\n'.format(dataset, str(e)))
    return bash_cmd