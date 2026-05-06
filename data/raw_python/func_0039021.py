def get_annotation(db_path, db_list):
    """ Checks if database is set as annotated. """

    annotated = False
    for db in db_list:
        if db["path"] == db_path:
            annotated = db["annotated"]
            break

    return annotated