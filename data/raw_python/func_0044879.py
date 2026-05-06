def ip_shell_after_exception(frame):
    """
    Launches an IPython embedded shell in the namespace where an exception occurred.

    :param frame:
    :return:
    """

    # let the user know, where this shell is 'waking up'
    # construct frame list
    # this will be printed in the header
    frame_info_list = []
    frame_list = []
    original_frame = frame = frame or inspect.currentframe()

    local_ns = frame.f_locals
    # global_ns = frame.f_globals  # this is deprecated by IPython
    dummy_module = DummyMod()
    dummy_module.__dict__ = frame.f_globals

    while frame is not None:
        frame_list.append(frame)
        info = inspect.getframeinfo(frame)
        frame_info_list.append(info)
        frame = frame.f_back

    frame_info_list.reverse()
    frame_info_str_list = [format_frameinfo(fi) for fi in frame_info_list]

    custom_header1 = "----- frame list -----\n\n"
    frame_info_str = "\n--\n".join(frame_info_str_list[:-1])
    custom_header2 = "\n----- ERROR -----\n"

    custom_header = "{0}{1}{2}".format(custom_header1, frame_info_str, custom_header2)

    # prevent IPython shell to be launched in IP-Notebook
    if len(frame_info_list) >= 2:
        test_str = str(frame_info_list[0]) + str(frame_info_list[1])
        if 'IPython' in test_str and 'zmq' in test_str:
            print("\n- Not entering IPython embedded shell  -\n")
            return

    # copied (and modified) from IPython/terminal/embed.py
    config = load_default_config()
    config.InteractiveShellEmbed = config.TerminalInteractiveShell

    # these two lines prevent problems in related to the initialization
    # of ultratb.FormattedTB below
    InteractiveShellEmbedWithoutBanner.clear_instance()
    InteractiveShellEmbedWithoutBanner._instance = None

    shell = InteractiveShellEmbedWithoutBanner.instance()

    shell(header=custom_header, stack_depth=2, local_ns=local_ns, module=dummy_module)

    # if `diff_index` is not None it will be interpreted as index increment for the frame_list in the except hook
    # "__mu" means "move up"
    diff_index = local_ns.get("__mu")
    if not isinstance(diff_index, int):
        diff_index = None

    return diff_index