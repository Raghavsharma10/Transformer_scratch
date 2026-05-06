def _handle_sigusr1(signum: int, frame: Any) -> None:
        """Print stacktrace."""
        print('=' * 70)
        print(''.join(traceback.format_stack()))
        print('-' * 70)