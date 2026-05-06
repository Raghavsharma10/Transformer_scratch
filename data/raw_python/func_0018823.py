def start_server(socket, projectname, xmlfilename: str) -> None:
    """Start the *HydPy* server using the given socket.

    The folder with the given `projectname` must be available within the
    current working directory.  The XML configuration file must be placed
    within the project folder unless `xmlfilename` is an absolute file path.
    The XML configuration file must be valid concerning the schema file
    `HydPyConfigMultipleRuns.xsd` (see method |ServerState.initialise|
    for further information).
    """
    state.initialise(projectname, xmlfilename)
    server = http.server.HTTPServer(('', int(socket)), HydPyServer)
    server.serve_forever()