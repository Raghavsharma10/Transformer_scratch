def main(xmpp_server, xmpp_port, peer_name, node_name, app_id,
         xmpp_jid=None, xmpp_password=None):
    """
    Runs the framework

    :param xmpp_server: Address of the XMPP server
    :param xmpp_port: Port of the XMPP server
    :param peer_name: Name of the peer
    :param node_name: Name (also, UID) of the node hosting the peer
    :param app_id: Application ID
    :param xmpp_jid: XMPP JID, None for Anonymous login
    :param xmpp_password: XMPP account password
    """
    # Create the framework
    framework = pelix.framework.create_framework(
        ('pelix.ipopo.core',
         'pelix.ipopo.waiting',
         'pelix.shell.core',
         'pelix.shell.ipopo',
         'pelix.shell.console',

         # Herald core
         'herald.core',
         'herald.directory',
         'herald.shell',

         # Herald XMPP
         'herald.transports.xmpp.directory',
         'herald.transports.xmpp.transport',

         # RPC
         'pelix.remote.dispatcher',
         'pelix.remote.registry',
         'herald.remote.discovery',
         'herald.remote.herald_xmlrpc',),
        {herald.FWPROP_NODE_UID: node_name,
         herald.FWPROP_NODE_NAME: node_name,
         herald.FWPROP_PEER_NAME: peer_name,
         herald.FWPROP_APPLICATION_ID: app_id})
    context = framework.get_bundle_context()

    # Start everything
    framework.start()

    # Instantiate components
    with use_waiting_list(context) as ipopo:
        # ... XMPP Transport
        ipopo.add(herald.transports.xmpp.FACTORY_TRANSPORT,
                  "herald-xmpp-transport",
                  {herald.transports.xmpp.PROP_XMPP_SERVER: xmpp_server,
                   herald.transports.xmpp.PROP_XMPP_PORT: xmpp_port,
                   herald.transports.xmpp.PROP_XMPP_JID: xmpp_jid,
                   herald.transports.xmpp.PROP_XMPP_PASSWORD: xmpp_password})

    # Start the framework and wait for it to stop
    framework.wait_for_stop()