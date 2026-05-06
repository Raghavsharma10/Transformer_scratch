def explanation(matchAmbiguous, concise, showLengths, showGaps, showNs):
    """
    Make an explanation of the output HTML table.

    @param matchAmbiguous: If C{True}, count ambiguous nucleotides that are
        possibly correct as actually being correct. Otherwise, we are strict
        and insist that only non-ambiguous nucleotides can contribute to the
        matching nucleotide count.
    @param concise: If C{True}, do not show match detail abbreviations.
    @param showLengths: If C{True}, include the lengths of sequences.
    @param showGaps: If C{True}, include the number of gaps in sequences.
    @param showNs: If C{True}, include the number of N characters in sequences.
    @return: A C{str} of HTML.
    """
    result = ["""
<h1>Sequence versus sequence identity table</h1>

<p>

The table cells below show the nucleotide identity fraction for the sequences
(<span class="best">like this</span> for the best value in each row). The
identity fraction numerator is the sum of the number of identical
    """]

    if matchAmbiguous:
        result.append('nucleotides plus the number of ambiguously matching '
                      'nucleotides.')
    else:
        result.append('nucleotides.')

    result.append("""The denominator
is the length of the sequence <em>for the row</em>. Sequence gaps
are not included when calculating their lengths.

</p>
    """)

    if showLengths or showGaps or showNs or matchAmbiguous or not concise:
        result.append("""
<p>

Key to abbreviations:
  <ul>
    """)

        if showLengths:
            result.append('<li>L: sequence Length.</li>')

        if showGaps:
            result.append('<li>G: number of Gaps in sequence.</li>')

        if showNs:
            result.append('<li>N: number of N characters in sequence.</li>')

        if not concise:
            result.append('<li>IM: Identical nucleotide Matches.</li>')

        if matchAmbiguous:
            result.append('<li>AM: Ambiguous nucleotide Matches.</li>')

        result.append("""
    <li>GG: Gap/Gap matches (both sequences have gaps).</li>
    <li>G?: Gap/Non-gap mismatches (one sequence has a gap).</li>
    <li>NE: Non-equal nucleotide mismatches.</li>
  </ul>
</p>
""")

    return '\n'.join(result)