def job_factory(self):
        """
        Create concrete jobs. The concrete jobs is following dictionary.
        jobs = {
            'PLUGINNAME-build_items': {
                'method': FUNCTION_OBJECT,
                'interval': INTERVAL_TIME ,
            }
            ...
        }
        If ConcreteJob instance has "build_discovery_items",
        "build_discovery_items" method is added to jobs.

        warn: looped method is deprecated in 0.4.0.
        You should implemente "build_items" instead of "looped_method".
        In most cases you need only to change the method name.
        """

        jobs = dict()

        for section, options in self.config.items():

            if section == 'global':
                continue

            # Since validate in utils/configread, does not occur here Error
            # In the other sections are global,
            # that there is a "module" option is collateral.
            plugin_name = options['module']
            job_kls = self.plugins[plugin_name]

            if hasattr(job_kls, '__init__'):
                job_argspec = inspect.getargspec(job_kls.__init__)

                if 'stats_queue' in job_argspec.args:
                    job_obj = job_kls(
                        options=options,
                        queue=self.queue,
                        stats_queue=self.stats_queue,
                        logger=self.logger
                    )

                else:
                    job_obj = job_kls(
                        options=options,
                        queue=self.queue,
                        logger=self.logger
                    )

            # Deprecated!!
            if hasattr(job_obj, 'looped_method'):
                self.logger.warn(
                    ('{0}\'s "looped_method" is deprecated.'
                     'Pleases change method name to "build_items"'
                     ''.format(plugin_name))
                )
                name = '-'.join([section, 'looped_method'])
                interval = 60
                if 'interval' in options:
                    interval = options['interval']
                elif 'interval' in self.config['global']:
                    interval = self.config['global']['interval']

                jobs[name] = {
                    'method': job_obj.looped_method,
                    'interval': interval,
                }

            if hasattr(job_obj, 'build_items'):
                name = '-'.join([section, 'build_items'])
                interval = 60
                if 'interval' in options:
                    interval = options['interval']
                elif 'interval' in self.config['global']:
                    interval = self.config['global']['interval']

                jobs[name] = {
                    'method': job_obj.build_items,
                    'interval': interval,
                }

                self.logger.info(
                    'load plugin {0} (interval {1})'
                    ''.format(plugin_name, interval)
                )

            if hasattr(job_obj, 'build_discovery_items'):
                name = '-'.join([section, 'build_discovery_items'])
                lld_interval = 600
                if 'lld_interval' in options:
                    lld_interval = options['lld_interval']
                elif 'lld_interval' in self.config['global']:
                    lld_interval = self.config['global']['lld_interval']

                jobs[name] = {
                    'method': job_obj.build_discovery_items,
                    'interval': lld_interval,
                }

                self.logger.info(
                    'load plugin {0} (lld_interval {1})'
                    ''.format(plugin_name, lld_interval)
                )

        return jobs