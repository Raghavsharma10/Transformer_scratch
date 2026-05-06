def getCollectDServer(queue, cfg):
    """Get the appropriate collectd server (multi processed or not)"""
    server = CollectDServerMP if cfg.collectd_workers > 1 else CollectDServer
    return server(queue, cfg)