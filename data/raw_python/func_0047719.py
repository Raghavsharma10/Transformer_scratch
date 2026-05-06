def get_effective_agent_id_with_proxy(proxy):
    """Given a Proxy, returns the Id of the effective Agent"""
    if is_authenticated_with_proxy(proxy):
        if proxy.has_effective_agent():
            return proxy.get_effective_agent_id()
        else:
            return proxy.get_authentication().get_agent_id()
    else:
        return Id(
            identifier='MC3GUE$T@MIT.EDU',
            namespace='authentication.Agent',
            authority='MIT-ODL')