def color_for_thread(thread_id):
    """
    Associates the thread ID with the next color in the `thread_colors` cycle,
    so that thread-specific parts of a log have a consistent separate color.
    """
    if thread_id not in seen_thread_colors:
        seen_thread_colors[thread_id] = next(thread_colors)

    return seen_thread_colors[thread_id]