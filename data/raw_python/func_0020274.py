def execute_script(self, name, keys, *args, **options):
        '''Execute a script.

        makes sure all required scripts are loaded.
        '''
        script = get_script(name)
        if not script:
            raise redis.RedisError('No such script "%s"' % name)
        address = self.address()
        if address not in all_loaded_scripts:
            all_loaded_scripts[address] = set()
        loaded = all_loaded_scripts[address]
        toload = script.required_scripts.difference(loaded)
        for name in toload:
            s = get_script(name)
            yield self.script_load(s.script)
        loaded.update(toload)
        yield script(self, keys, args, options)