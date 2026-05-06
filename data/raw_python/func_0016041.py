def modified_created(instance):
    """`modified` property must be later or equal to `created` property
    """
    if 'modified' in instance and 'created' in instance and \
            instance['modified'] < instance['created']:
        msg = "'modified' (%s) must be later or equal to 'created' (%s)"
        return JSONError(msg % (instance['modified'], instance['created']),
                         instance['id'])