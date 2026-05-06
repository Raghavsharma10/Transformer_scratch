def _create():
    """Globally called function for creating the isotope/element API."""
    def creator(group):
        """Helper function applied to each symbol group of the raw isotope table."""
        symbol = group['symbol'].values[0]
        try:    # Ghosts and custom atoms don't necessarily have an abundance fraction
            mass = (group['mass']*group['af']).sum()
            afm = group['af'].sum()
            if afm > 0.0:
                mass /= afm
        except ZeroDivisionError:
            mass = group['mass'].mean()
        znum = group['Z'].max()
        cov_radius = group['cov_radius'].mean()
        van_radius = group['van_radius'].mean()
        try:
            color = group.loc[group['af'].idxmax(), 'color']
        except TypeError:
            color = group['color'].values[0]
        name = group['name'].values[0]
        ele = Element(symbol, name, mass, znum, cov_radius, van_radius, color)
        # Attached isotopes
        for tope in group.apply(lambda s: Isotope(*s.tolist()), axis=1):
            setattr(ele, "_"+str(tope.A), tope)
        return ele

    iso = _rj(_E(_path).to_stream())
    iso.columns = _columns
    setattr(_this, "iso", iso)
    for element in iso.groupby("symbol").apply(creator):
        setattr(_this, element.symbol, element)