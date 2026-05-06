def get_web_drivers(cls, conf, global_capabilities=None):
        """Prepare 1 selenium driver instance per request browsers

        :param conf:
        :param global_capabilities:
        :return:
        """
        web_drivers = []
        if not global_capabilities:
            global_capabilities = {}
        else:
            global_capabilities = deepcopy(global_capabilities)
        grid_conf = deepcopy(conf)
        grid_conf.pop('class', None)
        request_drivers = grid_conf.pop('request_drivers', [])
        capabilities = grid_conf.pop('capabilities', {})
        global_capabilities.update(capabilities)
        for browser_req in request_drivers:
            name = 'grid'
            name = '%s_%s' % (name, browser_req.get('browserName'))
            name = '%s_%s' % (name, browser_req.get('version', 'lastest'))
            name = '%s_%s' % (name, browser_req.get('platform', 'ANY'))
            web_drivers.append(Grid(grid_conf, browser_req, name=name,
                                    global_capabilities=global_capabilities))
        return web_drivers