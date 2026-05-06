def stringClade(taxrefs, name, at):
    '''Return a Newick string from a list of TaxRefs'''
    string = []
    for ref in taxrefs:
        # distance is the difference between the taxonomic level of the ref
        #  and the current level of the tree growth
        d = float(at-ref.level)
        # ensure no spaces in ident, Newick tree cannot have spaces
        ident = re.sub("\s", "_", ref.ident)
        string.append('{0}:{1}'.format(ident, d))
    # join into single string with a name for the clade
    string = ','.join(string)
    string = '({0}){1}'.format(string, name)
    return string