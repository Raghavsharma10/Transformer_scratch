def _merge_layout(x: go.Layout, y: go.Layout) -> go.Layout:
    """Merge attributes from two layouts."""
    xjson = x.to_plotly_json()
    yjson = y.to_plotly_json()
    if 'shapes' in yjson and 'shapes' in xjson:
        xjson['shapes'] += yjson['shapes']
    yjson.update(xjson)
    return go.Layout(yjson)