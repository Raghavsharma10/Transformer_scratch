def _get_active_threads_names():
        """May contain sensitive info (like user ids). Use with care."""
        active_threads = threading.enumerate()
        return FormattedText().join(
            [
                FormattedText().newline().normal(" - {name}").start_format().bold(name=thread.name).end_format()
                for thread in active_threads
            ]
        )