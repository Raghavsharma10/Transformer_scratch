def WinMSGLoop():
    """Run the main windows message loop."""
    LPMSG = POINTER(MSG)
    LRESULT = c_ulong
    GetMessage = get_winfunc("user32", "GetMessageW", BOOL, (LPMSG, HWND, UINT, UINT))
    TranslateMessage = get_winfunc("user32", "TranslateMessage", BOOL, (LPMSG,))
    # restype = LRESULT
    DispatchMessage = get_winfunc("user32", "DispatchMessageW", LRESULT, (LPMSG,))

    msg = MSG()
    lpmsg = byref(msg)
    while GetMessage(lpmsg, HWND(), 0, 0) > 0:
        TranslateMessage(lpmsg)
        DispatchMessage(lpmsg)