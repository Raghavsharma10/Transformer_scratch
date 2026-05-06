def convertafield(field_comm, field_val, field_iddname):
    """convert field based on field info in IDD"""
    convinidd = ConvInIDD()
    field_typ = field_comm.get('type', [None])[0]    
    conv = convinidd.conv_dict().get(field_typ, convinidd.no_type)
    return conv(field_val, field_iddname)