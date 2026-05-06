def format_time(timestamp, time_format="%Y-%m-%d %H:%M:%S", never_placeholder="never", gmt=False):
    """strftime/localtime shortcut"""
    if not timestamp:
        return never_placeholder
    return time.strftime(time_format, time.gmtime(timestamp) if gmt else time.localtime(timestamp))