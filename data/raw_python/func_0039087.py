def update_roles_gce(use_cache=True, cache_expiration=86400, cache_path="~/.gcetools/instances", group_name=None, region=None, zone=None):
    """
    Dynamically update fabric's roles by using assigning the tags associated with
    each machine in Google Compute Engine.

    use_cache - will store a local cache in ~/.gcetools/
    cache_expiration - cache expiration in seconds (default: 1 day)
    cache_path - the path to store instances data (default: ~/.gcetools/instances)
    group_name - optional managed instance group to use instead of the global instance pool
    region - gce region name (such as `us-central1`) for a regional managed instance group
    zone - gce zone name (such as `us-central1-a`) for a zone managed instance group

    How to use:
    - Call 'update_roles_gce' at the end of your fabfile.py (it will run each
      time you run fabric).
    - On each function use the regular @roles decorator and set the role to the name
      of one of the tags associated with the instances you wish to work with
    """
    data = _get_data(use_cache, cache_expiration, group_name=group_name, region=region, zone=zone)
    roles = _get_roles(data)
    env.roledefs.update(roles)

    _data_loaded = True
    return INSTANCES_CACHE