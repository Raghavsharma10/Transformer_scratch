def six_frame(genome, table, minimum = 10):
    """
    translate each sequence into six reading frames
    """
    for seq in parse_fasta(genome):
        dna = Seq(seq[1].upper().replace('U', 'T'), IUPAC.ambiguous_dna)
        counter = 0
        for sequence in ['f', dna], ['rc', dna.reverse_complement()]:
            direction, sequence = sequence
            for frame in range(0, 3):
                for prot in \
                            sequence[frame:].\
                            translate(table = table, to_stop = False).split('*'):
                    if len(prot) < minimum:
                        continue
                    counter += 1
                    header = '%s_%s table=%s frame=%s-%s %s' % \
                                (seq[0].split()[0], counter, table, frame+1, \
                                direction, ' '.join(seq[0].split()[1:]))
                    yield [header, prot]