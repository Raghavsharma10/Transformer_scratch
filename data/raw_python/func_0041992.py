def update_state(world):
    """
    Increment the world state, determining which cells live, die, or appear.

    Args:
        world (list[list]): A square matrix of cells

    Returns: None
    """

    world_size = len(world)

    def wrap(index):
        """Wrap an index around the other end of the array"""
        return index % world_size

    for x in range(world_size):
        for y in range(world_size):
            # Decide if this node cares about the rules right now
            if not world[x][y].allow_change.get():
                continue
            live_neighbor_count = sum([
                world[wrap(x)][wrap(y + 1)].value,
                world[wrap(x + 1)][wrap(y + 1)].value,
                world[wrap(x + 1)][wrap(y)].value,
                world[wrap(x + 1)][wrap(y - 1)].value,
                world[wrap(x)][wrap(y-1)].value,
                world[wrap(x - 1)][wrap(y - 1)].value,
                world[wrap(x - 1)][wrap(y)].value,
                world[wrap(x - 1)][wrap(y + 1)].value
            ])
            if world[x][y].value:
                # Any live cell with fewer than two live neighbours dies
                # Any live cell with more than three live neighbours dies
                # Any live cell with two or three live neighbours lives
                if not (live_neighbor_count == 2 or live_neighbor_count == 3):
                    world[x][y].value = False
            else:
                # Any dead cell with exactly three live neighbours comes alive
                if live_neighbor_count == 3:
                    world[x][y].value = True