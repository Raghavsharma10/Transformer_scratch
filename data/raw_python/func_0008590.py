def clean_up(scripts):
    """Clean up the given list of scripts in-place so none of the scripts
    overlap.

    """
    scripts_with_pos = [s for s in scripts if s.pos]
    scripts_with_pos.sort(key=lambda s: (s.pos[1], s.pos[0]))
    scripts = scripts_with_pos + [s for s in scripts if not s.pos]

    y = 20
    for script in scripts:
        script.pos = (20, y)
        if isinstance(script, kurt.Script):
            y += stack_height(script.blocks)
        elif isinstance(script, kurt.Comment):
            y += 14
        y += 15