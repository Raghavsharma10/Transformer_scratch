def create_thread_color_cycle():
    """
    Generates a never-ending cycle of colors to choose from for individual
    threads.

    If color is not available, a cycle that repeats None every time is
    returned instead.
    """
    if not color_available:
        return itertools.cycle([None])

    return itertools.cycle(
        (
            colorama.Fore.CYAN,
            colorama.Fore.BLUE,
            colorama.Fore.MAGENTA,
            colorama.Fore.GREEN,
        )
    )