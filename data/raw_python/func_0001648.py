def _resolve_queue(self, queue, depth=0, links=None):
        """Finds the location of tej's queue directory on the server.

        The `queue` set when constructing this `RemoteQueue` might be relative
        to the home directory and might contain ``~user`` placeholders. Also,
        each queue may in fact be a link to another path (a file containing
        the string ``tejdir:``, a space, and a new pathname, relative to this
        link's location).
        """
        if depth == 0:
            logger.debug("resolve_queue(%s)", queue)
        answer = self.check_output(
            'if [ -d %(queue)s ]; then '
            '    cd %(queue)s; echo "dir"; cat version; pwd; '
            'elif [ -f %(queue)s ]; then '
            '    cat %(queue)s; '
            'else '
            '    echo no; '
            'fi' % {
                'queue': escape_queue(queue)})
        if answer == b'no':
            if depth > 0:
                logger.debug("Broken link at depth=%d", depth)
            else:
                logger.debug("Path doesn't exist")
            return None, depth
        elif answer.startswith(b'dir\n'):
            version, runtime, path = answer[4:].split(b'\n', 2)
            try:
                version = tuple(int(e)
                                for e in version.decode('ascii', 'ignore')
                                                .split('.'))
            except ValueError:
                version = 0, 0
            if version[:2] != self.PROTOCOL_VERSION:
                raise QueueExists(
                    msg="Queue exists and is using incompatible protocol "
                        "version %s" % '.'.join('%s' % e for e in version))
            path = PosixPath(path)
            runtime = runtime.decode('ascii', 'replace')
            if self.need_runtime is not None:
                if (self.need_runtime is not None and
                        runtime not in self.need_runtime):
                    raise QueueExists(
                        msg="Queue exists and is using explicitely disallowed "
                            "runtime %s" % runtime)
            logger.debug("Found directory at %s, depth=%d, runtime=%s",
                         path, depth, runtime)
            return path, depth
        elif answer.startswith(b'tejdir: '):
            new = queue.parent / answer[8:]
            logger.debug("Found link to %s, recursing", new)
            if links is not None:
                links.append(queue)
            return self._resolve_queue(new, depth + 1)
        else:  # pragma: no cover
            logger.debug("Server returned %r", answer)
            raise RemoteCommandFailure(msg="Queue resolution command failed "
                                           "in unexpected way")