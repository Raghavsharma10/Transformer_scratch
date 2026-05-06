def draw_canvas():
    """Render the tkinter canvas based on the state of ``world``"""
    for x in range(len(world)):
        for y in range(len(world[x])):
            if world[x][y].value:
                color = world[x][y].color_alive.get_as_hex()
            else:
                color = world[x][y].color_dead.get_as_hex()
            canvas.itemconfig(canvas_grid[x][y], fill=color)