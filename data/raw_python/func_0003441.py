def SSL_CTX_set_info_callback(ctx, app_info_cb):
    """
    Set the info callback

    :param callback: The Python callback to use
    :return: None
    """
    def py_info_callback(ssl, where, ret):
        try:
            app_info_cb(SSL(ssl), where, ret)
        except:
            pass
        return

    global _info_callback
    _info_callback[ctx] = _rvoid_voidp_int_int(py_info_callback)
    _SSL_CTX_set_info_callback(ctx, _info_callback[ctx])