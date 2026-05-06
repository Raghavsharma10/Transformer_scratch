def join_cols(cols):
    """Join list of columns into a string for a SQL query"""
    return ", ".join([i for i in cols]) if isinstance(cols, (list, tuple, set)) else cols