def _colorize_single_line(line, regexp, color_def):
    """Print single line to console with ability to colorize parts of it."""
    match = regexp.match(line)
    groupdict = match.groupdict()
    groups = match.groups()
    if not groupdict:
        # no named groups, just colorize whole line
        color = color_def[0]
        dark = color_def[1]
        cprint("%s\n" % line, color, fg_dark=dark)
    else:
        rev_groups = {v: k for k, v in groupdict.items()}
        for part in groups:
            if part in rev_groups and rev_groups[part] in color_def:
                group_name = rev_groups[part]
                cprint(
                    part,
                    color_def[group_name][0],
                    fg_dark=color_def[group_name][1],
                )
            else:
                cprint(part)
        cprint("\n")