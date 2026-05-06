def as_df():
    """Return a dataframe of isotopes."""
    records = []
    for sym, ele in vars(_this).items():
        if sym not in ["Element", "Isotope"] and not sym.startswith("_"):
            for k, v in vars(ele).items():
                if k.startswith("_") and k[1].isdigit():
                    records.append({kk: vv for kk, vv in vars(v).items() if not kk.startswith("_")})
    return _DF.from_records(records)