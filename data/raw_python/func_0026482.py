def construct_graph(args):
    """Preliminary HFOS application Launcher"""

    app = Core(args)

    setup_root(app)

    if args['debug']:
        from circuits import Debugger
        hfoslog("Starting circuits debugger", lvl=warn, emitter='GRAPH')
        dbg = Debugger().register(app)
        # TODO: Make these configurable from modules, navdata is _very_ noisy
        # but should not be listed _here_
        dbg.IgnoreEvents.extend([
            "read", "_read", "write", "_write",
            "stream_success", "stream_complete",
            "serial_packet", "raw_data", "stream",
            "navdatapush", "referenceframe",
            "updateposition", "updatesubscriptions",
            "generatevesseldata", "generatenavdata", "sensordata",
            "reset_flood_offenders", "reset_flood_counters",  # Flood counters
            "task_success", "task_done",  # Thread completion
            "keepalive"  # IRC Gateway
        ])

    hfoslog("Beginning graph assembly.", emitter='GRAPH')

    if args['drawgraph']:
        from circuits.tools import graph

        graph(app)

    if args['opengui']:
        import webbrowser
        # TODO: Fix up that url:
        webbrowser.open("http://%s:%i/" % (args['host'], args['port']))

    hfoslog("Graph assembly done.", emitter='GRAPH')

    return app