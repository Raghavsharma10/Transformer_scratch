async def set_utc(pyvlx):
    """Enable house status monitor."""
    setutc = SetUTC(pyvlx=pyvlx)
    await setutc.do_api_call()
    if not setutc.success:
        raise PyVLXException("Unable to set utc.")