async def house_status_monitor_disable(pyvlx):
    """Disable house status monitor."""
    status_monitor_disable = HouseStatusMonitorDisable(pyvlx=pyvlx)
    await status_monitor_disable.do_api_call()
    if not status_monitor_disable.success:
        raise PyVLXException("Unable disable house status monitor.")