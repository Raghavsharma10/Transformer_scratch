def setup_path(invoke_minversion=None):
    """Setup python search and add ``TASKS_VENDOR_DIR`` (if available)."""
    # print("INVOKE.tasks: setup_path")
    if not os.path.isdir(TASKS_VENDOR_DIR):
        print("SKIP: TASKS_VENDOR_DIR=%s is missing" % TASKS_VENDOR_DIR)
        return
    elif os.path.abspath(TASKS_VENDOR_DIR) in sys.path:
        # -- SETUP ALREADY DONE:
        # return
        pass

    use_vendor_bundles = os.environ.get("INVOKE_TASKS_USE_VENDOR_BUNDLES", "no")
    if need_vendor_bundles(invoke_minversion):
        use_vendor_bundles = "yes"

    if use_vendor_bundles == "yes":
        syspath_insert(0, os.path.abspath(TASKS_VENDOR_DIR))
        if setup_path_for_bundle(INVOKE_BUNDLE, pos=1):
            import invoke
            bundle_path = os.path.relpath(INVOKE_BUNDLE, os.getcwd())
            print("USING: %s (version: %s)" % (bundle_path, invoke.__version__))
    else:
        # -- BEST-EFFORT: May rescue something
        syspath_append(os.path.abspath(TASKS_VENDOR_DIR))
        setup_path_for_bundle(INVOKE_BUNDLE, pos=len(sys.path))

    if DEBUG_SYSPATH:
        for index, p in enumerate(sys.path):
            print("  %d.  %s" % (index, p))