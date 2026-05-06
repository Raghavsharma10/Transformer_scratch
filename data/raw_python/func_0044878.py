def IPS(copy_namespaces=True, overwrite_globals=False):
    """Starts IPython embedded shell. This is similar to IPython.embed() but with some
    additional features:

    1. Print a list of the calling frames before entering the prompt
    2. (optionally) copy local name space to global one to prevent certain IPython bug.
    3. while doing so optinally overwrite names in the global namespace

    """

    # let the user know, where this shell is 'waking up'
    # construct frame list
    # this will be printed in the header
    frame_info_list = []
    frame_list = []
    frame = inspect.currentframe()
    while frame is not None:
        frame_list.append(frame)
        info = inspect.getframeinfo(frame)
        frame_info_list.append(info)
        frame = frame.f_back

    frame_info_list.reverse()
    frame_list.reverse()
    frame_info_str_list = [format_frameinfo(fi) for fi in frame_info_list]

    custom_header1 = "----- frame list -----\n\n"
    frame_info_str = "\n--\n".join(frame_info_str_list[:-1])
    custom_header2 = "\n----- end of frame list -----\n"

    custom_header = "{0}{1}{2}".format(custom_header1, frame_info_str, custom_header2)

    # prevent IPython shell to be launched in IP-Notebook
    test_str = str(frame_info_list[0]) + str(frame_info_list[1])
    if 'IPython' in test_str and 'zmq' in test_str:
        print("\n- Not entering IPython embedded shell  -\n")
        return

    # copied (and modified) from IPython/terminal/embed.py
    config = load_default_config()

    config.InteractiveShellEmbed = config.TerminalInteractiveShell

    # these two lines prevent problems related to the initialization
    # of ultratb.FormattedTB below
    InteractiveShellEmbed.clear_instance()
    InteractiveShellEmbed._instance = None

    shell = InteractiveShellEmbed.instance()

    # achieve that custom macros are loade in interactive shell
    shell.magic('load_ext storemagic')
    if config.StoreMagics.autorestore:
        shell.magic('store -r')
        ar_keys = [k.split("/")[-1] for k in shell.db.keys() if k.startswith("autorestore/")]
    else:
        ar_keys = []

    # adapt the namespaces to prevent missing names inside the shell
    # see: https://github.com/ipython/ipython/issues/62
    # https://github.com/ipython/ipython/issues/10695
    if copy_namespaces and len(frame_list) >= 2:
        # callers_frame to IPS()
        # note that frame_list and frame_info_list were reversed above
        f1 = frame_list[-2]
        lns = f1.f_locals
        gns = f1.f_globals

        l_keys = set(lns)
        g_keys = set(gns)
        u_keys = shell.user_ns.keys()

        # those keys which are in local ns but not in global
        safe_keys = l_keys - g_keys
        unsafe_keys = l_keys.intersection(g_keys)

        assert safe_keys.union(unsafe_keys) == l_keys

        gns.update({k:lns[k] for k in safe_keys})

        if unsafe_keys and not overwrite_globals:
            custom_header += "following local keys have " \
                             "not been copied:\n{}\n".format(unsafe_keys)

        if unsafe_keys and overwrite_globals:
            gns.update({k:lns[k] for k in unsafe_keys})
            custom_header += "following global keys have " \
                             "been overwritten:\n{}\n".format(unsafe_keys)

        # now update the gns with stuff from the user_ns (if it will not overwrite anything)
        # this could be implemented cleaner
        for k in ar_keys:
            if k not in gns:
                gns[k] = shell.user_ns[k]
            else:
                print("omitting key from user_namespace:", k)

        dummy_module = DummyMod()
        dummy_module.__dict__ = gns

    else:
        # unexpected few frames or no copying desired:
        lns = None
        dummy_module = None

    # now execute the shell
    shell(header=custom_header, stack_depth=2, local_ns=lns, module=dummy_module)

    custom_excepthook = getattr(sys, 'custom_excepthook', None)
    if custom_excepthook is not None:
        assert callable(custom_excepthook)
        sys.excepthook = custom_excepthook