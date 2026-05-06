def get_agent(msg):
    """ Handy hack to handle legacy messages where 'agent' was a list.  """
    agent = msg['msg']['agent']
    if isinstance(agent, list):
        agent = agent[0]
    return agent