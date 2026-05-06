def to_prettytable(df):
    """Convert DataFrame into ``PrettyTable``.
    """
    pt = PrettyTable()
    pt.field_names = df.columns
    for tp in zip(*(l for col, l in df.iteritems())):
        pt.add_row(tp)
    return pt