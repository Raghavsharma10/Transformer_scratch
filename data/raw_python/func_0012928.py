def makeductbranch(idf, bname):
    """make a branch with a duct
    use standard inlet outlet names"""
    # make the duct component first
    pname = "%s_duct" % (bname,)
    aduct = makeductcomponent(idf, pname)
    # now make the branch with the duct in it
    abranch = idf.newidfobject("BRANCH", Name=bname)
    abranch.Component_1_Object_Type = 'duct'
    abranch.Component_1_Name = pname
    abranch.Component_1_Inlet_Node_Name = aduct.Inlet_Node_Name
    abranch.Component_1_Outlet_Node_Name = aduct.Outlet_Node_Name
    abranch.Component_1_Branch_Control_Type = "Bypass"
    return abranch