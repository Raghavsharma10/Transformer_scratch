def ckan_db_init(self, retry_seconds=DB_INIT_RETRY_SECONDS):
        """
        Run db init to create all ckan tables

        :param retry_seconds: how long to retry waiting for db to start
        """
        # XXX workaround for not knowing how long we need to wait
        # for postgres to be ready. fix this by changing the postgres
        # entrypoint, or possibly running once with command=/bin/true
        started = time.time()
        while True:
            try:
                self.run_command(
                    '/usr/lib/ckan/bin/paster --plugin=ckan db init '
                    '-c /project/development.ini',
                    db_links=True,
                    clean_up=True,
                    )
                break
            except WebCommandError:
                if started + retry_seconds > time.time():
                    raise
            time.sleep(DB_INIT_RETRY_DELAY)