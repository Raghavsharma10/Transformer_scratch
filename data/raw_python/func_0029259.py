def get_receive(self, script_list):
        """Return a list of received events contained in script_list."""
        events = defaultdict(set)
        for script in script_list:
            if self.script_start_type(script) == self.HAT_WHEN_I_RECEIVE:
                event = script.blocks[0].args[0].lower()
                events[event].add(script)
        return events