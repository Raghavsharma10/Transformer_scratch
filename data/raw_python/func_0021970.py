def _get_worker_pools_names(worker_pools: list):
        """May contain sensitive info (like user ids). Use with care."""
        return FormattedText().join(
            [
                FormattedText().newline().normal(" - {name}").start_format().bold(name=worker.name).end_format()
                for worker in worker_pools
            ]
        )