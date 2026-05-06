def create_temp_space():
    """
    Create a new temporary cloud foundry space for
    a project.
    """
    # Truncating uuid to just take final 12 characters since space name
    # is used to name services and there is a 50 character limit on instance
    # names.  
    # MAINT: hacky with possible collisions
    unique_name = str(uuid.uuid4()).split('-')[-1]
    admin = predix.admin.cf.spaces.Space()
    res = admin.create_space(unique_name)

    space = predix.admin.cf.spaces.Space(
            guid=res['metadata']['guid'],
            name=res['entity']['name'])
    space.target()

    return space