def bounding_box(dset):
    '''return the coordinates (in RAI) of the corners of a box enclosing the data in ``dset``'''
    o = nl.run(["3dAutobox","-input",dset])
    ijk_coords = re.findall(r'[xyz]=(\d+)\.\.(\d+)',o.output)
    from_rai = ijk_to_xyz(dset,[float(x[0]) for x in ijk_coords])
    to_rai = ijk_to_xyz(dset,[float(x[1]) for x in ijk_coords])
    return (from_rai,to_rai)