def hashes_above(path, line_number):
    """Yield hashes from contiguous comment lines before line ``line_number``.

    """
    def hash_lists(path):
        """Yield lists of hashes appearing between non-comment lines.

        The lists will be in order of appearance and, for each non-empty
        list, their place in the results will coincide with that of the
        line number of the corresponding result from `parse_requirements`
        (which changed in pip 7.0 to not count comments).

        """
        hashes = []
        with open(path) as file:
            for lineno, line in enumerate(file, 1):
                match = HASH_COMMENT_RE.match(line)
                if match:  # Accumulate this hash.
                    hashes.append(match.groupdict()['hash'])
                if not IGNORED_LINE_RE.match(line):
                    yield hashes  # Report hashes seen so far.
                    hashes = []
                elif PIP_COUNTS_COMMENTS:
                    # Comment: count as normal req but have no hashes.
                    yield []

    return next(islice(hash_lists(path), line_number - 1, None))