def build_jar(source_jar_path, dest_jar_path, config, venv=None, definition=None, logdir=None):
    """Build a StormTopology .jar which encapsulates the topology defined in
    topology_dir. Optionally override the module and function names. This
    feature supports the definition of multiple topologies in a single
    directory."""

    if definition is None:
        definition = 'create.create'

    # Prepare data we'll use later for configuring parallelism.
    config_yaml = read_yaml(config)
    parallelism = dict((k.split('.')[-1], v) for k, v in six.iteritems(config_yaml)
        if k.startswith('petrel.parallelism'))

    pip_options = config_yaml.get('petrel.pip_options', '')

    module_name, dummy, function_name = definition.rpartition('.')
    
    topology_dir = os.getcwd()

    # Make a copy of the input "jvmpetrel" jar. This jar acts as a generic
    # starting point for all Petrel topologies.
    source_jar_path = os.path.abspath(source_jar_path)
    dest_jar_path = os.path.abspath(dest_jar_path)
    if source_jar_path == dest_jar_path:
        raise ValueError("Error: Destination and source path are the same.")
    shutil.copy(source_jar_path, dest_jar_path)
    jar = zipfile.ZipFile(dest_jar_path, 'a', compression=zipfile.ZIP_DEFLATED)
    
    added_path_entry = False
    try:
        # Add the files listed in manifest.txt to the jar.
        with open(os.path.join(topology_dir, MANIFEST), 'r') as f:
            for fn in f.readlines():
                # Ignore blank and comment lines.
                fn = fn.strip()
                if len(fn) and not fn.startswith('#'):

                    add_item_to_jar(jar, os.path.expandvars(fn.strip()))

        # Add user and machine information to the jar.
        add_to_jar(jar, '__submitter__.yaml', '''
petrel.user: %s
petrel.host: %s
''' % (getpass.getuser(),socket.gethostname()))
        
        # Also add the topology configuration to the jar.
        with open(config, 'r') as f:
            config_text = f.read()
        add_to_jar(jar, '__topology__.yaml', config_text)
    
        # Call module_name/function_name to populate a Thrift topology object.
        builder = TopologyBuilder()
        module_dir = os.path.abspath(topology_dir)
        if module_dir not in sys.path:
            sys.path[:0] = [ module_dir ]
            added_path_entry = True
        module = __import__(module_name)
        getattr(module, function_name)(builder)

        # Add the spout and bolt Python scripts to the jar. Create a
        # setup_<script>.sh for each Python script.

        # Add Python scripts and any other per-script resources.
        for k, v in chain(six.iteritems(builder._spouts), six.iteritems(builder._bolts)):
            add_file_to_jar(jar, topology_dir, v.script)

            # Create a bootstrap script.
            if venv is not None:
                # Allow overriding the execution command from the "petrel"
                # command line. This is handy if the server already has a
                # virtualenv set up with the necessary libraries.
                v.execution_command = os.path.join(venv, 'bin/python')

            # If a parallelism value was specified in the configuration YAML,
            # override any setting provided in the topology definition script.
            if k in parallelism:
                builder._commons[k].parallelism_hint = int(parallelism.pop(k))

            v.execution_command, v.script = \
                intercept(venv, v.execution_command, os.path.splitext(v.script)[0],
                          jar, pip_options, logdir)

        if len(parallelism):
            raise ValueError(
                'Parallelism settings error: There are no components named: %s' %
                ','.join(parallelism.keys()))

        # Build the Thrift topology object and serialize it to the .jar. Must do
        # this *after* the intercept step above since that step may modify the
        # topology definition.
        buf = io.BytesIO()
        builder.write(buf)
        add_to_jar(jar, 'topology.ser', buf.getvalue())
    finally:
        jar.close()
        if added_path_entry:
            # Undo our sys.path change.
            sys.path[:] = sys.path[1:]