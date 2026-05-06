def makeductcomponent(idf, dname):
    """make a duct component
    generate inlet outlet names"""
    aduct = idf.newidfobject("duct".upper(), Name=dname)
    aduct.Inlet_Node_Name = "%s_inlet" % (dname,)
    aduct.Outlet_Node_Name = "%s_outlet" % (dname,)
    return aduct