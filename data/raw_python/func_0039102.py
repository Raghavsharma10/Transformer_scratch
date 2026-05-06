def files(patterns,
          require_tags=("require",),
          include_tags=("include",),
          exclude_tags=("exclude",),
          root=".",
          always_exclude=("**/.git*", "**/.lfs*", "**/.c9*", "**/.~c9*")):
    """
    Takes a list of lib50._config.TaggedValue returns which files should be included and excluded from `root`.
    Any pattern tagged with a tag
        from include_tags will be included
        from require_tags can only be a file, that will then be included. MissingFilesError is raised if missing
        from exclude_tags will be excluded
    Any pattern in always_exclude will always be excluded.
    """
    require_tags = list(require_tags)
    include_tags = list(include_tags)
    exclude_tags = list(exclude_tags)

    # Ensure every tag starts with !
    for tags in [require_tags, include_tags, exclude_tags]:
        for i, tag in enumerate(tags):
            tags[i] = tag if tag.startswith("!") else "!" + tag

    with cd(root):
        # Include everything by default
        included = _glob("*")
        excluded = set()

        if patterns:
            missing_files = []

            # Per line in files
            for pattern in patterns:
                # Include all files that are tagged with !require
                if pattern.tag in require_tags:
                    file = str(Path(pattern.value))
                    if not Path(file).exists():
                        missing_files.append(file)
                    else:
                        try:
                            excluded.remove(file)
                        except KeyError:
                            pass
                        else:
                            included.add(file)
                # Include all files that are tagged with !include
                elif pattern.tag in include_tags:
                    new_included = _glob(pattern.value)
                    excluded -= new_included
                    included.update(new_included)
                # Exclude all files that are tagged with !exclude
                elif pattern.tag in exclude_tags:
                    new_excluded = _glob(pattern.value)
                    included -= new_excluded
                    excluded.update(new_excluded)

            if missing_files:
                raise MissingFilesError(missing_files)

    # Exclude all files that match a pattern from always_exclude
    for line in always_exclude:
        included -= _glob(line)

    # Exclude any files that are not valid utf8
    invalid = set()
    for file in included:
        try:
            file.encode("utf8")
        except UnicodeEncodeError:
            excluded.add(file.encode("utf8", "replace").decode())
            invalid.add(file)
    included -= invalid

    return included, excluded