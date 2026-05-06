def find(s):
    """
    Find an amino acid whose name or abbreviation is s.

    @param s: A C{str} amino acid specifier. This may be a full name,
        a 3-letter abbreviation or a 1-letter abbreviation. Case is ignored.
    return: An L{AminoAcid} instance or C{None} if no matching amino acid can
        be located.
    """

    abbrev1 = None
    origS = s

    if ' ' in s:
        # Convert first word to title case, others to lower.
        first, rest = s.split(' ', 1)
        s = first.title() + ' ' + rest.lower()
    else:
        s = s.title()

    if s in NAMES:
        abbrev1 = s
    elif s in ABBREV3_TO_ABBREV1:
        abbrev1 = ABBREV3_TO_ABBREV1[s]
    elif s in NAMES_TO_ABBREV1:
        abbrev1 = NAMES_TO_ABBREV1[s]
    else:
        # Look for a 3-letter codon.
        def findCodon(target):
            for abbrev1, codons in CODONS.items():
                for codon in codons:
                    if codon == target:
                        return abbrev1

        abbrev1 = findCodon(origS.upper())

    if abbrev1:
        return AminoAcid(
            NAMES[abbrev1], ABBREV3[abbrev1], abbrev1, CODONS[abbrev1],
            PROPERTIES[abbrev1], PROPERTY_DETAILS[abbrev1],
            PROPERTY_CLUSTERS[abbrev1])