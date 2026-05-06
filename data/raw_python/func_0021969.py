def _get_running_workers_names(running_workers: list):
        """May contain sensitive info (like user ids). Use with care."""
        return FormattedText().join(
            [
                FormattedText().newline().normal(" - {name}").start_format().bold(name=worker.name).end_format()
                for worker in running_workers
            ]
        )