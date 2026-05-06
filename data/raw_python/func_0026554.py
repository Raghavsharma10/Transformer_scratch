def create_module(clear_target, target):
    """Creates a new template HFOS plugin module"""

    if os.path.exists(target):
        if clear_target:
            shutil.rmtree(target)
        else:
            log("Target exists! Use --clear to delete it first.",
                emitter='MANAGE')
            sys.exit(2)

    done = False
    info = None

    while not done:
        info = _ask_questionnaire()
        pprint(info)
        done = _ask('Is the above correct', default='y', data_type='bool')

    augmented_info = _augment_info(info)

    log("Constructing module %(plugin_name)s" % info)
    _construct_module(augmented_info, target)