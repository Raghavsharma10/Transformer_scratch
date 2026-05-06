def replacebranch(idf, loop, branch,
                  listofcomponents, fluid=None,
                  debugsave=False,
                  testing=None):
    """It will replace the components in the branch with components in
    listofcomponents"""
    if fluid is None:
        fluid = ''
    # -------- testing ---------
    testn = 0
    # -------- testing ---------

    # join them into a branch
    # -----------------------
    # np1_inlet -> np1 -> np1_np2_node -> np2 -> np2_outlet
        # change the node names in the component
        # empty the old branch
        # fill in the new components with the node names into this branch
    listofcomponents = _clean_listofcomponents(listofcomponents)

    components = [item[0] for item in listofcomponents]
    connectcomponents(idf, listofcomponents, fluid=fluid)
    if debugsave:
        idf.savecopy("hhh3.idf")
    # -------- testing ---------
    testn = doingtesting(testing, testn)
    if testn == None:
        returnnone()
    # -------- testing ---------
    fields = SomeFields.a_fields

    thebranch = branch
    componentsintobranch(idf, thebranch, listofcomponents, fluid=fluid)
    if debugsave:
        idf.savecopy("hhh4.idf")
    # -------- testing ---------
    testn = doingtesting(testing, testn)
    if testn == None:
        returnnone()
    # -------- testing ---------

    # # gather all renamed nodes
    # # do the renaming
    renamenodes(idf, 'node')
    if debugsave:
        idf.savecopy("hhh7.idf")
    # -------- testing ---------
    testn = doingtesting(testing, testn)
    if testn == None:
        returnnone()
    # -------- testing ---------

    # check for the end nodes of the loop
    if loop.key == 'AIRLOOPHVAC':
        fields = SomeFields.a_fields
    if loop.key == 'PLANTLOOP':
        fields = SomeFields.p_fields
    if loop.key == 'CONDENSERLOOP':
        fields = SomeFields.c_fields
    # for use in bunch
    flnames = [field.replace(' ', '_') for field in fields]

    if fluid.upper() == 'WATER':
        supplyconlistname = loop[flnames[3]]
        # Plant_Side_Connector_List_Name or Condenser_Side_Connector_List_Name
    elif fluid.upper() == 'AIR':
        supplyconlistname = loop[flnames[1]]  # Connector_List_Name'
    supplyconlist = idf.getobject('CONNECTORLIST', supplyconlistname)
    for i in range(1, 100000):  # large range to hit end
        try:
            fieldname = 'Connector_%s_Object_Type' % (i,)
            ctype = supplyconlist[fieldname]
        except bunch_subclass.BadEPFieldError:
            break
        if ctype.strip() == '':
            break
        fieldname = 'Connector_%s_Name' % (i,)
        cname = supplyconlist[fieldname]
        connector = idf.getobject(ctype.upper(), cname)
        if connector.key == 'CONNECTOR:SPLITTER':
            firstbranchname = connector.Inlet_Branch_Name
            cbranchname = firstbranchname
            isfirst = True
        if connector.key == 'CONNECTOR:MIXER':
            lastbranchname = connector.Outlet_Branch_Name
            cbranchname = lastbranchname
            isfirst = False
        if cbranchname == thebranch.Name:
            # rename end nodes
            comps = getbranchcomponents(idf, thebranch)
            if isfirst:
                comp = comps[0]
                inletnodename = getnodefieldname(
                    comp,
                    "Inlet_Node_Name", fluid)
                comp[inletnodename] = [
                    comp[inletnodename],
                    loop[flnames[0]]]  # Plant_Side_Inlet_Node_Name
            else:
                comp = comps[-1]
                outletnodename = getnodefieldname(
                    comp,
                    "Outlet_Node_Name", fluid)
                comp[outletnodename] = [
                    comp[outletnodename],
                    loop[flnames[1]]]  # .Plant_Side_Outlet_Node_Name
    # -------- testing ---------
    testn = doingtesting(testing, testn)
    if testn == None:
        returnnone()
    # -------- testing ---------

    if fluid.upper() == 'WATER':
        demandconlistname = loop[flnames[7]]  # .Demand_Side_Connector_List_Name
        demandconlist = idf.getobject('CONNECTORLIST', demandconlistname)
        for i in range(1, 100000):  # large range to hit end
            try:
                fieldname = 'Connector_%s_Object_Type' % (i,)
                ctype = demandconlist[fieldname]
            except bunch_subclass.BadEPFieldError:
                break
            if ctype.strip() == '':
                break
            fieldname = 'Connector_%s_Name' % (i,)
            cname = demandconlist[fieldname]
            connector = idf.getobject(ctype.upper(), cname)
            if connector.key == 'CONNECTOR:SPLITTER':
                firstbranchname = connector.Inlet_Branch_Name
                cbranchname = firstbranchname
                isfirst = True
            if connector.key == 'CONNECTOR:MIXER':
                lastbranchname = connector.Outlet_Branch_Name
                cbranchname = lastbranchname
                isfirst = False
            if cbranchname == thebranch.Name:
                # rename end nodes
                comps = getbranchcomponents(idf, thebranch)
                if isfirst:
                    comp = comps[0]
                    inletnodename = getnodefieldname(
                        comp,
                        "Inlet_Node_Name", fluid)
                    comp[inletnodename] = [
                        comp[inletnodename],
                        loop[flnames[4]]]  # .Demand_Side_Inlet_Node_Name
                if not isfirst:
                    comp = comps[-1]
                    outletnodename = getnodefieldname(
                        comp,
                        "Outlet_Node_Name", fluid)
                    comp[outletnodename] = [
                        comp[outletnodename],
                        loop[flnames[5]]]  # .Demand_Side_Outlet_Node_Name

    # -------- testing ---------
    testn = doingtesting(testing, testn)
    if testn == None:
        returnnone()
    # -------- testing ---------

    if debugsave:
        idf.savecopy("hhh8.idf")

    # # gather all renamed nodes
    # # do the renaming
    renamenodes(idf, 'node')
    # -------- testing ---------
    testn = doingtesting(testing, testn)
    if testn == None:
        returnnone()
    # -------- testing ---------
    if debugsave:
        idf.savecopy("hhh9.idf")
    return thebranch