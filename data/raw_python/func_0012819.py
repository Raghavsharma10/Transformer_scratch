def _make_ntgrid(grid):
    """make a named tuple grid

    [["",  "a b", "b c", "c d"],
     ["x y", 1,     2,     3 ],
     ["y z", 4,     5,     6 ],
     ["z z", 7,     8,     9 ],]
    will return
    ntcol(x_y=ntrow(a_b=1, b_c=2, c_d=3),
          y_z=ntrow(a_b=4, b_c=5, c_d=6),
          z_z=ntrow(a_b=7, b_c=8, c_d=9))"""
    hnames = [_nospace(n) for n in grid[0][1:]]
    vnames = [_nospace(row[0]) for row in grid[1:]]
    vnames_s = " ".join(vnames)
    hnames_s = " ".join(hnames)
    ntcol = collections.namedtuple('ntcol', vnames_s)
    ntrow = collections.namedtuple('ntrow', hnames_s)
    rdict = [dict(list(zip(hnames, row[1:]))) for row in grid[1:]]
    ntrows = [ntrow(**rdict[i]) for i, name in enumerate(vnames)]
    ntcols = ntcol(**dict(list(zip(vnames, ntrows))))
    return ntcols