def remove_lines(lines, remove=('[[back to top]', '<a class="mk-toclify"')):
    """Removes existing [back to top] links and <a id> tags."""

    if not remove:
        return lines[:]

    out = []
    for l in lines:
        if l.startswith(remove):
            continue
        out.append(l)
    return out