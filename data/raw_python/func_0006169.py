def execute_commands(self, mapping, *args, **kwargs):
        """Concurrently executes a sequence of commands on a Redis cluster that
        are associated with a routing key, returning a new mapping where
        values are a list of results that correspond to the command in the same
        position. For example::

            >>> cluster.execute_commands({
            ...   'foo': [
            ...     ('PING',),
            ...     ('TIME',),
            ...   ],
            ...   'bar': [
            ...     ('CLIENT', 'GETNAME'),
            ...   ],
            ... })
            {'bar': [<Promise None>],
             'foo': [<Promise True>, <Promise (1454446079, 418404)>]}

        Commands that are instances of :class:`redis.client.Script` will first
        be checked for their existence on the target nodes then loaded on the
        targets before executing and can be interleaved with other commands::

            >>> from redis.client import Script
            >>> TestScript = Script(None, 'return {KEYS, ARGV}')
            >>> cluster.execute_commands({
            ...   'foo': [
            ...     (TestScript, ('key:1', 'key:2'), range(0, 3)),
            ...   ],
            ...   'bar': [
            ...     (TestScript, ('key:3', 'key:4'), range(3, 6)),
            ...   ],
            ... })
            {'bar': [<Promise [['key:3', 'key:4'], ['3', '4', '5']]>],
             'foo': [<Promise [['key:1', 'key:2'], ['0', '1', '2']]>]}

        Internally, :class:`FanoutClient` is used for issuing commands.
        """
        def is_script_command(command):
            return isinstance(command[0], Script)

        def check_script_load_result(script, result):
            if script.sha != result:
                raise AssertionError(
                    'Hash mismatch loading {!r}: expected {!r}, got {!r}'.format(
                        script,
                        script.sha,
                        result,
                    )
                )

        # Run through all the commands and check to see if there are any
        # scripts, and whether or not they have been loaded onto the target
        # hosts.
        exists = {}
        with self.fanout(*args, **kwargs) as client:
            for key, commands in mapping.items():
                targeted = client.target_key(key)
                for command in filter(is_script_command, commands):
                    script = command[0]

                    # Set the script hash if it hasn't already been set.
                    if not script.sha:
                        script.sha = sha1(script.script).hexdigest()

                    # Check if the script has been loaded on each host that it
                    # will be executed on.
                    for host in targeted._target_hosts:
                        if script not in exists.setdefault(host, {}):
                            exists[host][script] = targeted.execute_command('SCRIPT EXISTS', script.sha)

        # Execute the pending commands, loading scripts onto servers where they
        # do not already exist.
        results = {}
        with self.fanout(*args, **kwargs) as client:
            for key, commands in mapping.items():
                results[key] = []
                targeted = client.target_key(key)
                for command in commands:
                    # If this command is a script, we need to check and see if
                    # it needs to be loaded before execution.
                    if is_script_command(command):
                        script = command[0]
                        for host in targeted._target_hosts:
                            if script in exists[host]:
                                result = exists[host].pop(script)
                                if not result.value[0]:
                                    targeted.execute_command('SCRIPT LOAD', script.script).done(
                                        on_success=functools.partial(check_script_load_result, script)
                                    )
                        keys, arguments = command[1:]
                        parameters = list(keys) + list(arguments)
                        results[key].append(targeted.execute_command('EVALSHA', script.sha, len(keys), *parameters))
                    else:
                        results[key].append(targeted.execute_command(*command))

        return results