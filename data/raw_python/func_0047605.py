def run(m, w, trace=False, steps=1000, show_stack=3):
    """Runs an automaton, automatically selecting a search method."""

    # Check to see whether run_pda can handle it.
    is_pda = True
    stack = None
    if not m.oneway:
        is_pda = False
    for s in range(m.num_stores):
        if s == m.input:
            pass
        elif m.has_cell(s): # anything with finite number of configs would do
            pass
        elif m.has_stack(s):
            if stack is None:
                stack = s
            else:
                is_pda = False
        else:
            is_pda = False

    if is_pda and stack is not None:
        if trace: print("using modified Lang algorithm")
        return run_pda(m, w, stack=stack, trace=trace, show_stack=show_stack)
    else:
        if trace: print("using breadth-first search")
        return run_bfs(m, w, trace=trace, steps=steps)