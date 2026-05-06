def setup_json_capture(osbs, os_conf, capture_dir):
    """
    Only used for setting up the testing framework.
    """

    try:
        os.mkdir(capture_dir)
    except OSError:
        pass
    finally:
        osbs.os._con.request = ResponseSaver(capture_dir,
                                             os_conf.get_openshift_api_uri(),
                                             os_conf.get_k8s_api_uri(),
                                             osbs.os._con.request).request