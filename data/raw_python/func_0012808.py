def wallinterzone(idf, bsdobject, deletebsd=True, setto000=False):
    """return an wall:interzone object if the bsd (buildingsurface:detailed) 
    is an interaone wall"""
    # ('WALL:INTERZONE', Wall, Surface OR Zone OR OtherSideCoefficients)
    # test if it is an exterior wall
    if bsdobject.Surface_Type.upper() == 'WALL': # Surface_Type == wall
        if bsdobject.Outside_Boundary_Condition.upper() in ('SURFACE', 'ZONE', 'OtherSideCoefficients'.upper()): 
            simpleobject = idf.newidfobject('WALL:INTERZONE')
            simpleobject.Name = bsdobject.Name
            simpleobject.Construction_Name = bsdobject.Construction_Name
            simpleobject.Zone_Name = bsdobject.Zone_Name
            obco = 'Outside_Boundary_Condition_Object'
            simpleobject[obco] = bsdobject[obco]
            simpleobject.Azimuth_Angle = bsdobject.azimuth
            simpleobject.Tilt_Angle = bsdobject.tilt
            surforigin = bsdorigin(bsdobject, setto000=setto000)
            simpleobject.Starting_X_Coordinate = surforigin[0]
            simpleobject.Starting_Y_Coordinate = surforigin[1]
            simpleobject.Starting_Z_Coordinate = surforigin[2]
            simpleobject.Length = bsdobject.width
            simpleobject.Height = bsdobject.height
            if deletebsd:
                idf.removeidfobject(bsdobject)
            return simpleobject
    return None