def get_env_key(obj, key=None):
    """
    Return environment variable key to use for lookups within a
    namespace represented by the package name.

    For example, any varialbes for predix.security.uaa are stored
    as PREDIX_SECURITY_UAA_KEY
    """
    return str.join('_', [obj.__module__.replace('.','_').upper(),
        key.upper()])