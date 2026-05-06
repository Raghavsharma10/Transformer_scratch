def dePrefixAndSuffixFasta(sequences):
    """
    sequences: an iterator producing Bio.Seq sequences.

    return: a generator of sequences with no duplicates and no fully contained
        subsequences.
    """
    sequences = sorted(sequences, key=lambda s: len(s.seq), reverse=True)
    seen = set()
    for s in sequences:
        thisSeq = str(s.seq)
        thisHash = md5(thisSeq.encode('UTF-8')).digest()
        if thisHash not in seen:
            # Add prefixes.
            newHash = md5()
            for nucl in thisSeq:
                newHash.update(nucl.encode('UTF-8'))
                seen.add(newHash.digest())
            # Add suffixes.
            for start in range(len(thisSeq) - 1):
                seen.add(md5(thisSeq[start + 1:].encode('UTF-8')).digest())
            yield s