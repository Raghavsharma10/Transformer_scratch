async def house_status_monitor_enable(pyvlx):
    """Enable house status monitor."""
    status_monitor_enable = HouseStatusMonitorEnable(pyvlx=pyvlx)
    await status_monitor_enable.do_api_call()
    if not status_monitor_enable.success:
        raise PyVLXException("Unable enable house status monitor.")