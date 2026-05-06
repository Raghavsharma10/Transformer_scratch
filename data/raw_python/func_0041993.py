def click_event(event):
    """On click, bring the cell under the cursor to Life"""
    grid_x_coord = int(divmod(event.x, cell_size)[0])
    grid_y_coord = int(divmod(event.y, cell_size)[0])
    world[grid_x_coord][grid_y_coord].value = True
    color = world[x][y].color_alive.get_as_hex()
    canvas.itemconfig(canvas_grid[grid_x_coord][grid_y_coord], fill=color)