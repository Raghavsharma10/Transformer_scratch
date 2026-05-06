def makepipebranch(idf, bname):
    """make a branch with a pipe
    use standard inlet outlet names"""
    # make the pipe component first
    pname = "%s_pipe" % (bname,)
    apipe = makepipecomponent(idf, pname)
    # now make the branch with the pipe in it
    abranch = idf.newidfobject("BRANCH", Name=bname)
    abranch.Component_1_Object_Type = 'Pipe:Adiabatic'
    abranch.Component_1_Name = pname
    abranch.Component_1_Inlet_Node_Name = apipe.Inlet_Node_Name
    abranch.Component_1_Outlet_Node_Name = apipe.Outlet_Node_Name
    abranch.Component_1_Branch_Control_Type = "Bypass"
    return abranch