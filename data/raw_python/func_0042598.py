def random_board(max_x, max_y, load_factor):
    """Return a random board with given max x and y coords."""
    return dict(((randint(0, max_x), randint(0, max_y)), 0) for _ in
                xrange(int(max_x * max_y / load_factor)))