def read_csv(filename):
    """Reads a CSV file containing a tabular description of a transition function,
       as found in Sipser. Major difference: instead of multiple header rows,
       only a single header row whose entries might be tuples.
       """

    with open(filename) as file:
        table = list(csv.reader(file))
    m = from_table(table)
    return m