def getDescsV2(flags, fs_list=(), hs_list=(), ss_list=(), os_list=()):
    """
    Return a FunctionFS descriptor suitable for serialisation.

    flags (int)
        Any combination of VIRTUAL_ADDR, EVENTFD, ALL_CTRL_RECIP,
        CONFIG0_SETUP.
    {fs,hs,ss,os}_list (list of descriptors)
        Instances of the following classes:
        {fs,hs,ss}_list:
            USBInterfaceDescriptor
            USBEndpointDescriptorNoAudio
            USBEndpointDescriptor
            USBSSEPCompDescriptor
            USBSSPIsocEndpointDescriptor
            USBOTGDescriptor
            USBOTG20Descriptor
            USBInterfaceAssocDescriptor
            TODO: HID
            All (non-empty) lists must define the same number of interfaces
            and endpoints, and endpoint descriptors must be given in the same
            order, bEndpointAddress-wise.
        os_list:
            OSDesc
    """
    count_field_list = []
    descr_field_list = []
    kw = {}
    for descriptor_list, flag, prefix, allowed_descriptor_klass in (
        (fs_list, HAS_FS_DESC, 'fs', USBDescriptorHeader),
        (hs_list, HAS_HS_DESC, 'hs', USBDescriptorHeader),
        (ss_list, HAS_SS_DESC, 'ss', USBDescriptorHeader),
        (os_list, HAS_MS_OS_DESC, 'os', OSDescHeader),
    ):
        if descriptor_list:
            for index, descriptor in enumerate(descriptor_list):
                if not isinstance(descriptor, allowed_descriptor_klass):
                    raise TypeError(
                        'Descriptor %r of unexpected type: %r' % (
                            index,
                            type(descriptor),
                        ),
                    )
            descriptor_map = [
                ('desc_%i' % x, y)
                for x, y in enumerate(descriptor_list)
            ]
            flags |= flag
            count_name = prefix + 'count'
            descr_name = prefix + 'descr'
            count_field_list.append((count_name, le32))
            descr_type = type(
                't_' + descr_name,
                (ctypes.LittleEndianStructure, ),
                {
                    '_pack_': 1,
                    '_fields_': [
                        (x, type(y))
                        for x, y in descriptor_map
                    ],
                }
            )
            descr_field_list.append((descr_name, descr_type))
            kw[count_name] = len(descriptor_map)
            kw[descr_name] = descr_type(**dict(descriptor_map))
        elif flags & flag:
            raise ValueError(
                'Flag %r set but descriptor list empty, cannot generate type.' % (
                    FLAGS.get(flag),
                )
            )
    klass = type(
        'DescsV2_0x%02x' % (
            flags & (
                HAS_FS_DESC |
                HAS_HS_DESC |
                HAS_SS_DESC |
                HAS_MS_OS_DESC
            ),
            # XXX: include contained descriptors type information ? (and name ?)
        ),
        (DescsHeadV2, ),
        {
            '_fields_': count_field_list + descr_field_list,
        },
    )
    return klass(
        magic=DESCRIPTORS_MAGIC_V2,
        length=ctypes.sizeof(klass),
        flags=flags,
        **kw
    )