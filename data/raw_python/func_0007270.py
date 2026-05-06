def slugify_headline(line, remove_dashes=False):
    """
    Takes a header line from a Markdown document and
    returns a tuple of the
        '#'-stripped version of the head line,
        a string version for <a id=''></a> anchor tags,
        and the level of the headline as integer.
    E.g.,
    >>> dashify_headline('### some header lvl3')
    ('Some header lvl3', 'some-header-lvl3', 3)

    """
    stripped_right = line.rstrip('#')
    stripped_both = stripped_right.lstrip('#')
    level = len(stripped_right) - len(stripped_both)
    stripped_wspace = stripped_both.strip()

    # character replacements
    replaced_colon = stripped_wspace.replace('.', '')
    replaced_slash = replaced_colon.replace('/', '')
    rem_nonvalids = ''.join([c if c in VALIDS
                             else '-' for c in replaced_slash])

    lowered = rem_nonvalids.lower()
    slugified = re.sub(r'(-)\1+', r'\1', lowered)  # remove duplicate dashes
    slugified = slugified.strip('-')  # strip dashes from start and end

    # exception '&' (double-dash in github)
    slugified = slugified.replace('-&-', '--')

    if remove_dashes:
        slugified = slugified.replace('-','')

    return [stripped_wspace, slugified, level]