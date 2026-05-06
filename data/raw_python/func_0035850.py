def generate_admin_metadata(name, creator_username=None):
    """Return admin metadata as a dictionary."""

    if not dtoolcore.utils.name_is_valid(name):
        raise(DtoolCoreInvalidNameError())

    if creator_username is None:
        creator_username = dtoolcore.utils.getuser()

    datetime_obj = datetime.datetime.utcnow()

    admin_metadata = {
        "uuid": str(uuid.uuid4()),
        "dtoolcore_version": __version__,
        "name": name,
        "type": "protodataset",
        "creator_username": creator_username,
        "created_at": dtoolcore.utils.timestamp(datetime_obj)
    }
    return admin_metadata