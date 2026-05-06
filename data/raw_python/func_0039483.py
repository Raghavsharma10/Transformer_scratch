def main(http_port, peer_name, node_name, app_id):
    """
    Runs the framework

    :param http_port: HTTP port to listen to
    :param peer_name: Name of the peer
    :param node_name: Name (also, UID) of the node hosting the peer
    :param app_id: Application ID
    """
    # Create the framework
    framework = pelix.framework.create_framework(
        ('pelix.ipopo.core',
         'pelix.ipopo.waiting',
         'pelix.shell.core',
         'pelix.shell.ipopo',
         'pelix.shell.console',
         'pelix.http.basic',

         # Herald core
         'herald.core',
         'herald.directory',
         'herald.shell',

         # Herald HTTP
         'herald.transports.http.directory',
         'herald.transports.http.discovery_multicast',
         'herald.transports.http.servlet',
         'herald.transports.http.transport',

         # RPC
         'pelix.remote.dispatcher',
         'pelix.remote.registry',
         'herald.remote.discovery',
         'herald.remote.herald_xmlrpc',),
        {herald.FWPROP_NODE_UID: node_name,
         herald.FWPROP_NODE_NAME: node_name,
         herald.FWPROP_PEER_NAME: peer_name,
         herald.FWPROP_APPLICATION_ID: app_id})

    # Start everything
    framework.start()
    context = framework.get_bundle_context()

    # Instantiate components
    with use_waiting_list(context) as ipopo:
        # ... HTTP server
        ipopo.add(pelix.http.FACTORY_HTTP_BASIC, "http-server",
                  {pelix.http.HTTP_SERVICE_PORT: http_port})

        # ... HTTP reception servlet
        ipopo.add(herald.transports.http.FACTORY_SERVLET,
                  "herald-http-servlet")

        # ... HTTP multicast discovery
        ipopo.add(herald.transports.http.FACTORY_DISCOVERY_MULTICAST,
                  "herald-http-discovery-multicast")

    # Start the framework and wait for it to stop
    framework.wait_for_stop()