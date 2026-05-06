def provision_system_user(items, database_name, overwrite=False, clear=False, skip_user_check=False):
    """Provision a system user"""

    from hfos.provisions.base import provisionList
    from hfos.database import objectmodels

    # TODO: Add a root user and make sure owner can access it later.
    # Setting up details and asking for a password here is not very useful,
    # since this process is usually run automated.

    if overwrite is True:
        hfoslog('Refusing to overwrite system user!', lvl=warn,
                emitter='PROVISIONS')
        overwrite = False

    system_user_count = objectmodels['user'].count({'name': 'System'})
    if system_user_count == 0 or clear is False:
        provisionList(Users, 'user', overwrite, clear,  skip_user_check=True)
        hfoslog('Provisioning: Users: Done.', emitter="PROVISIONS")
    else:
        hfoslog('System user already present.', lvl=warn, emitter='PROVISIONS')