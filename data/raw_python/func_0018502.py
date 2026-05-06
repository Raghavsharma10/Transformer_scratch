def default_combine_filenames_generator(filenames, max_length=40):
    """Return a new filename to use as the combined file name for a
    bunch of files.
    A precondition is that they all have the same file extension

    Given that the list of files can have different paths, we aim to use the
    most common path.

    Example:
      /somewhere/else/foo.js
      /somewhere/bar.js
      /somewhere/different/too/foobar.js
    The result will be
      /somewhere/foo_bar_foobar.js

    Another thing to note, if the filenames have timestamps in them, combine
    them all and use the highest timestamp.

    """
    path = None
    names = []
    extension = None
    timestamps = []
    for filename in filenames:
        name = os.path.basename(filename)
        if not extension:
            extension = os.path.splitext(name)[1]
        elif os.path.splitext(name)[1] != extension:
            raise ValueError("Can't combine multiple file extensions")

        for each in re.finditer('\.\d{10}\.', name):
            timestamps.append(int(each.group().replace('.','')))
            name = name.replace(each.group(), '.')
        name = os.path.splitext(name)[0]
        names.append(name)

        if path is None:
            path = os.path.dirname(filename)
        else:
            if len(os.path.dirname(filename)) < len(path):
                path = os.path.dirname(filename)


    new_filename = '_'.join(names)
    if timestamps:
        new_filename += ".%s" % max(timestamps)

    new_filename = new_filename[:max_length]
    new_filename += extension

    return os.path.join(path, new_filename)