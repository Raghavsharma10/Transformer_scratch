def analyze(self, scratch, **kwargs):
        """Run and return the results from the BroadcastReceive plugin."""
        all_scripts = list(self.iter_scripts(scratch))
        results = defaultdict(set)
        broadcast = dict((x, self.get_broadcast_events(x))  # Events by script
                         for x in all_scripts)
        correct = self.get_receive(all_scripts)
        results['never broadcast'] = set(correct.keys())

        for script, events in broadcast.items():
            for event in events.keys():
                if event is True:  # Remove dynamic broadcasts
                    results['dynamic broadcast'].add(script.morph.name)
                    del events[event]
                elif event in correct:
                    results['never broadcast'].discard(event)
                else:
                    results['never received'].add(event)

        # remove events from correct dict that were never broadcast
        for event in correct.keys():
            if event in results['never broadcast']:
                del correct[event]

        # Find scripts that have more than one broadcast event on any possible
        # execution path through the program
        # TODO: Permit mutually exclusive broadcasts
        for events in broadcast.values():
            if len(events) > 1:
                for event in events:
                    if event in correct:
                        results['parallel broadcasts'].add(event)
                        del correct[event]

        # Find events that have two (or more) receivers in which one of the
        # receivers has a "delay" block
        for event, scripts in correct.items():
            if len(scripts) > 1:
                for script in scripts:
                    for _, _, block in self.iter_blocks(script.blocks):
                        if block.type.shape == 'stack':
                            results['multiple receivers with delay'].add(event)
                            if event in correct:
                                del correct[event]

        results['success'] = set(correct.keys())
        return {'broadcast': results}