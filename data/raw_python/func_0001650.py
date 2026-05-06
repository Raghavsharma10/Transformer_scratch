def setup(self, links=None, force=False, only_links=False):
        """Installs the runtime at the target location.

        This will not replace an existing installation, unless `force` is True.

        After installation, creates links to this installation at the specified
        locations.
        """
        if not links:
            links = []

        if only_links:
            logger.info("Only creating links")
            for link in links:
                self.check_call('echo "tejdir:" %(queue)s > %(link)s' % {
                                'queue': escape_queue(self.queue),
                                'link': escape_queue(link)})
            return

        queue, depth = self._resolve_queue(self.queue)
        if queue is not None or depth > 0:
            if force:
                if queue is None:
                    logger.info("Replacing broken link")
                elif depth > 0:
                    logger.info("Replacing link to %s...", queue)
                else:
                    logger.info("Replacing existing queue...")
                self.check_call('rm -Rf %s' % escape_queue(self.queue))
            else:
                if queue is not None and depth > 0:
                    raise QueueExists("Queue already exists (links to %s)\n"
                                      "Use --force to replace" % queue)
                elif depth > 0:
                    raise QueueExists("Broken link exists\n"
                                      "Use --force to replace")
                else:
                    raise QueueExists("Queue already exists\n"
                                      "Use --force to replace")

        queue = self._setup()

        for link in links:
            self.check_call('echo "tejdir:" %(queue)s > %(link)s' % {
                'queue': escape_queue(queue),
                'link': escape_queue(link)})