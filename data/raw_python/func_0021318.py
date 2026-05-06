def graph_format(new_mem, old_mem, is_firstiteration=True):
    """Show changes graphically in memory consumption"""
    if is_firstiteration:
        output = "  n/a   "
    elif new_mem - old_mem > 50000000:
        output = "   +++++"
    elif new_mem - old_mem > 20000000:
        output = "   ++++ "
    elif new_mem - old_mem > 5000000:
        output = "   +++  "
    elif new_mem - old_mem > 1000000:
        output = "   ++   "
    elif new_mem - old_mem > 50000:
        output = "   +    "
    elif old_mem - new_mem > 10000000:
        output = "---     "
    elif old_mem - new_mem > 2000000:
        output = " --     "
    elif old_mem - new_mem > 100000:
        output = "  -     "
    else:
        output = "        "
    return output