def rotate_concurrent(self, *locations, **kw):
        """
        Rotate the backups in the given locations concurrently.

        :param locations: One or more values accepted by :func:`coerce_location()`.
        :param kw: Any keyword arguments are passed on to :func:`rotate_backups()`.

        This function uses :func:`rotate_backups()` to prepare rotation
        commands for the given locations and then it removes backups in
        parallel, one backup per mount point at a time.

        The idea behind this approach is that parallel rotation is most useful
        when the files to be removed are on different disks and so multiple
        devices can be utilized at the same time.

        Because mount points are per system :func:`rotate_concurrent()` will
        also parallelize over backups located on multiple remote systems.
        """
        timer = Timer()
        pool = CommandPool(concurrency=10)
        logger.info("Scanning %s ..", pluralize(len(locations), "backup location"))
        for location in locations:
            for cmd in self.rotate_backups(location, prepare=True, **kw):
                pool.add(cmd)
        if pool.num_commands > 0:
            backups = pluralize(pool.num_commands, "backup")
            logger.info("Preparing to rotate %s (in parallel) ..", backups)
            pool.run()
            logger.info("Successfully rotated %s in %s.", backups, timer)