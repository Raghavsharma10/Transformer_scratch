def update_monitors(self):
        """
        Periodically check in with Reflex Engine and refresh the list of what to monitor
        """
        self.thread_debug("Starting monitor refresh", module="update_monitors")

        # need to make a more efficient way of doing this via Reflex Engine
        monitors = []
        self.rcs.cache_reset()

        svcs = self.rcs.cache_list('service',
                                   cols=['pipeline', 'name',
                                         'active-instances'])
        for svc in svcs:
            try:
                pipeline = self.rcs.cache_get('pipeline', svc['pipeline'])
                for mon in pipeline.get('monitor', []):
                    self.DEBUG("monitor {}".format(mon))
                    mon['service'] = svc['name']
                    mon['pipeline'] = svc['pipeline']
                    for inst_name in svc.get('active-instances', []):
                        inst = self.rcs.cache_get('instance', inst_name)

                        # todo: insert: macro flatten

                        mymon = mon.copy()
                        mymon['instance'] = inst_name
                        mymon['target'] = inst['address']
                        mymon['title'] = svc['name'] + ": " + mon['name']
                        monitors.append(mymon)
            except KeyboardInterrupt:
                raise
            except: # pylint: disable=bare-except
                self.NOTIFY("Error in processing monitor:", err=traceback.format_exc())

        self.NOTIFY("Refreshed monitors", total_monitors=len(monitors))
        self.DEBUG("Monitors", monitors=monitors)

        # mutex / threadsafe?
        self.monitors = monitors
        cache = self.rcs._cache # pylint: disable=protected-access
        self.instances = cache['instance']
        self.services = cache['service']
        self.pipelines = cache['pipeline']
        self.thread_debug("Refresh complete", module="update_monitors")