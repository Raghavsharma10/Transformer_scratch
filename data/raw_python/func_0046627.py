def host_context(func):
    "Sets the context of the setting to the current host"
    @wraps(func)
    def decorator(*args, **kwargs):
        hosts = get_hosts_settings()
        with settings(**hosts[env.host]):
            return func(*args, **kwargs)
    return decorator