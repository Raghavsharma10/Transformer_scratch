def human_play(w, i, grid):
    "Just ask for a move."
    plaint = ''
    prompt = whose_move(grid) + " move? [1-9] "
    while True:
        w.render_to_terminal(w.array_from_text(view(grid)
                                               + '\n\n' + plaint + prompt))
        key = c = i.next()
        try:
            move = int(key)
        except ValueError:
            pass
        else:
            if 1 <= move <= 9:
                successor = apply_move(grid, from_human_move(move))
                if successor: return successor
        plaint = ("Hey, that's illegal. Give me one of these digits:\n\n"
                  + (grid_format
                     % tuple(move if apply_move(grid, from_human_move(move)) else '-'
                             for move in range(1, 10))
                     + '\n\n'))