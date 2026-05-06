def print_progress_bar(text, done, total, width):
    """
    Print progress bar.
    """
    if total > 0:
        n = int(float(width) * float(done) / float(total))
        sys.stdout.write("\r{0} [{1}{2}] ({3}/{4})".format(text, '#' * n, ' ' * (width - n), done, total))
        sys.stdout.flush()