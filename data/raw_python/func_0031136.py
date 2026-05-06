def basePlotter(blastHits, title):
    """
    Plot the reads and the subject, so that bases in the reads which are
    different from the subject are shown. Else a '.' is shown.
    like so:
    subject_gi  ATGCGTACGTACGACACC
    read_1         A......TTC..T

    @param blastHits: A L{dark.blast.BlastHits} instance.
    @param title: A C{str} sequence title that was matched by BLAST. We plot
        the reads that matched this title.
    """
    result = []
    params = blastHits.plotParams
    assert params is not None, ('Oops, it looks like you forgot to run '
                                'computePlotInfo.')

    sequence = ncbidb.getSequence(title, blastHits.records.blastDb)
    subject = sequence.seq
    gi = title.split('|')[1]
    sub = '%s\t \t \t%s' % (gi, subject)
    result.append(sub)

    plotInfo = blastHits.titles[title]['plotInfo']
    assert plotInfo is not None, ('Oops, it looks like you forgot to run '
                                  'computePlotInfo.')

    items = plotInfo['items']
    count = 0
    for item in items:
        count += 1
        hsp = item['hsp']
        queryTitle = blastHits.fasta[item['readNum']].id
        # If the product of the subject and query frame values is +ve,
        # then they're either both +ve or both -ve, so we just use the
        # query as is. Otherwise, we need to reverse complement it.
        if item['frame']['subject'] * item['frame']['query'] > 0:
            query = blastHits.fasta[item['readNum']].seq
            reverse = False
        else:
            # One of the subject or query has negative sense.
            query = blastHits.fasta[
                item['readNum']].reverse_complement().seq
            reverse = True
        query = query.upper()
        queryStart = hsp['queryStart']
        subjectStart = hsp['subjectStart']
        queryEnd = hsp['queryEnd']
        subjectEnd = hsp['subjectEnd']

        # Before comparing the read to the subject, make a string of the
        # same length as the subject, which contains the read and
        # has ' ' where the read does not match.
        # 3 parts need to be taken into account:
        # 1) the left offset (if the query doesn't stick out to the left)
        # 2) the query. if the frame is -1, it has to be reversed.
        # The query consists of 3 parts: left, middle (control for gaps)
        # 3) the right offset

        # Do part 1) and 2).
        if queryStart < 0:
            # The query is sticking out to the left.
            leftQuery = ''
            if subjectStart == 0:
                # The match starts at the first base of the subject.
                middleLeftQuery = ''
            else:
                # The match starts into the subject.
                # Determine the length of the not matching query
                # part to the left.
                leftOffset = -1 * queryStart
                rightOffset = subjectStart + leftOffset
                middleLeftQuery = query[leftOffset:rightOffset]
        else:
            # The query is not sticking out to the left
            # make the left offset.
            leftQuery = queryStart * ' '

            leftQueryOffset = subjectStart - queryStart
            middleLeftQuery = query[:leftQueryOffset]

        # Do part 3).
        # Disregard gaps in subject while adding.
        matchQuery = item['origHsp'].query
        matchSubject = item['origHsp'].sbjct
        index = 0
        mid = ''
        for item in range(len(matchQuery)):
            if matchSubject[index] != ' ':
                mid += matchQuery[index]
            index += 1
        # if the query has been reversed, turn the matched part around
        if reverse:
            rev = ''
            toReverse = mid
            reverseDict = {' ': ' ', '-': '-', 'A': 'T', 'T': 'A',
                           'C': 'G', 'G': 'C', '.': '.', 'N': 'N'}
            for item in toReverse:
                newItem = reverseDict[item]
                rev += newItem
            mid = rev[::-1]

        middleQuery = middleLeftQuery + mid

        # add right not-matching part of the query
        rightQueryOffset = queryEnd - subjectEnd
        rightQuery = query[-rightQueryOffset:]
        middleQuery += rightQuery

        read = leftQuery + middleQuery

        # do part 3)
        offset = len(subject) - len(read)
        # if the read is sticking out to the right
        # chop it off
        if offset < 0:
            read = read[:offset]
        # if it's not sticking out, fill the space with ' '
        elif offset > 0:
            read += offset * ' '

        # compare the subject and the read, make a string
        # called 'comparison', which contains a '.' if the bases
        # are equal and the letter of the read if they are not.
        comparison = ''
        for readBase, subjectBase in zip(read, subject):
            if readBase == ' ':
                comparison += ' '
            elif readBase == subjectBase:
                comparison += '.'
            elif readBase != subjectBase:
                comparison += readBase
            index += 1
        que = '%s \t %s' % (queryTitle, comparison)
        result.append(que)

        # sanity checks
        assert (len(comparison) == len(subject)), (
            '%d != %d' % (len(comparison), len(subject)))

        index = 0
        if comparison[index] == ' ':
            index += 1
        else:
            start = index - 1
            assert (start == queryStart or start == -1), (
                '%s != %s or %s != -1' % (start, queryStart, start))

    return result