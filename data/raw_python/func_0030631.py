def warehouse_query(line, cell):
    "my cell magic"
    from IPython import get_ipython

    parts = line.split()
    w_var_name = parts.pop(0)
    w = get_ipython().ev(w_var_name)

    w.query(cell).close()