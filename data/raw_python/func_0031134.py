def load_fixtures(self, nodes=None, progress_callback=None, dry_run=False):
        """Load all fixtures for given nodes.

        If no nodes are given all fixtures will be loaded.
        :param list nodes: list of nodes to be loaded.
        :param callable progress_callback: Callback which will be called while
                                           handling the nodes.
        """

        if progress_callback and not callable(progress_callback):
            raise Exception('Callback should be callable')

        plan = self.get_plan(nodes=nodes)

        try:
            with transaction.atomic():
                self.load_plan(plan=plan, progress_callback=progress_callback)
                if dry_run:
                    raise DryRun
        except DryRun:
            # Dry-run to get the atomic transaction rolled back
            pass

        return len(plan)