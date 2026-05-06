def countPrint(mesg, count, len1, len2=None):
    """
    Format a message followed by an integer count and a percentage (or
    two, if the sequence lengths are unequal).

    @param mesg: a C{str} message.
    @param count: a numeric value.
    @param len1: the C{int} length of sequence 1.
    @param len2: the C{int} length of sequence 2. If C{None}, will
        default to C{len1}.
    @return: A C{str} for printing.
    """
    def percentage(a, b):
        """
        What percent of a is b?

        @param a: a numeric value.
        @param b: a numeric value.
        @return: the C{float} percentage.
        """
        return 100.0 * a / b if b else 0.0

    if count == 0:
        return '%s: %d' % (mesg, count)
    else:
        len2 = len2 or len1
        if len1 == len2:
            return '%s: %d/%d (%.2f%%)' % (
                mesg, count, len1, percentage(count, len1))
        else:
            return ('%s: %d/%d (%.2f%%) of sequence 1, '
                    '%d/%d (%.2f%%) of sequence 2' % (
                        mesg,
                        count, len1, percentage(count, len1),
                        count, len2, percentage(count, len2)))