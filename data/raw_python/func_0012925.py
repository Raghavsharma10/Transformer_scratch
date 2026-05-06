def makepipecomponent(idf, pname):
    """make a pipe component
    generate inlet outlet names"""
    apipe = idf.newidfobject("Pipe:Adiabatic".upper(), Name=pname)
    apipe.Inlet_Node_Name = "%s_inlet" % (pname,)
    apipe.Outlet_Node_Name = "%s_outlet" % (pname,)
    return apipe