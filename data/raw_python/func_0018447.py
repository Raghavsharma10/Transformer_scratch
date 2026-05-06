def get_ssh_agent_keys(logger):
    """
    Ask the SSH agent for a list of keys, and return it.

    :return: A reference to the SSH agent and a list of keys.
    """
    agent, agent_keys = None, None

    try:
        agent = paramiko.agent.Agent()
        _agent_keys = agent.get_keys()

        if not _agent_keys:
            agent.close()
            logger.error(
                "SSH agent didn't provide any valid key. Trying to continue..."
            )
        else:
            agent_keys = tuple(k for k in _agent_keys)
    except paramiko.SSHException:
        if agent:
            agent.close()
            agent = None
        logger.error("SSH agent speaks a non-compatible protocol. Ignoring it.")
    finally:
        return agent, agent_keys