def tictactoe(w, i, player, opponent, grid=None):
    "Put two strategies to a classic battle of wits."
    grid = grid or empty_grid
    while True:
        w.render_to_terminal(w.array_from_text(view(grid)))
        if is_won(grid):
            print(whose_move(grid), "wins.")
            break
        if not successors(grid):
            print("A draw.")
            break
        grid = player(w, i, grid)
        player, opponent = opponent, player