def default(inst):
    """Default routine to be applied when loading data. Removes redundant naming

    """
    import pysat.instruments.icon_ivm as icivm
    inst.tag = 'level_2'
    icivm.remove_icon_names(inst, target='ICON_L2_EUV_Daytime_OP_')