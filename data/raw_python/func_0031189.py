def main(gi, ranges):
    """
    Print the features of the genbank entry given by gi. If ranges is
    non-emtpy, only print features that include the ranges.

    gi: either a hit from a BLAST record, in the form
        'gi|63148399|gb|DQ011818.1|' or a gi number (63148399 in this example).
    ranges: a possibly empty list of ranges to print information for. Each
        range is a non-descending (start, end) pair of integers.
    """
    # TODO: Make it so we can pass a 'db' argument to getSequence.
    record = getSequence(gi)

    if record is None:
        print("Looks like you're offline.")
        sys.exit(3)
    else:
        printed = set()
        if ranges:
            for (start, end) in ranges:
                for index, feature in enumerate(record.features):
                    if (start < int(feature.location.end) and
                            end > int(feature.location.start) and
                            index not in printed):
                        print(feature)
                        printed.add(index)
        else:
            # Print all features.
            for feature in record.features:
                print(feature)