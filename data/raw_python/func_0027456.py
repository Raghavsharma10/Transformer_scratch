def common_values_dict():
    """Build a basic values object used in every create method.

       All our resources contain a same subset of value. Instead of
       redoing this code everytime, this method ensures it is done only at
       one place.
    """
    now = datetime.datetime.utcnow().isoformat()
    etag = utils.gen_etag()
    values = {
        'id': utils.gen_uuid(),
        'created_at': now,
        'updated_at': now,
        'etag': etag
    }

    return values