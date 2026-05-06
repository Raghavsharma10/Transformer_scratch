def getInterfaceInAllSpeeds(interface, endpoint_list, class_descriptor_list=()):
    """
    Produce similar fs, hs and ss interface and endpoints descriptors.
    Should be useful for devices desiring to work in all 3 speeds with maximum
    endpoint wMaxPacketSize. Reduces data duplication from descriptor
    declarations.
    Not intended to cover fancy combinations.

    interface (dict):
      Keyword arguments for
        getDescriptor(USBInterfaceDescriptor, ...)
      in all speeds.
      bNumEndpoints must not be provided.
    endpoint_list (list of dicts)
      Each dict represents an endpoint, and may contain the following items:
      - "endpoint": required, contains keyword arguments for
          getDescriptor(USBEndpointDescriptorNoAudio, ...)
        or
          getDescriptor(USBEndpointDescriptor, ...)
        The with-audio variant is picked when its extra fields are assigned a
        value.
        wMaxPacketSize may be missing, in which case it will be set to the
        maximum size for given speed and endpoint type.
        bmAttributes must be provided.
        If bEndpointAddress is zero (excluding direction bit) on the first
        endpoint, endpoints will be assigned their rank in this list,
        starting at 1. Their direction bit is preserved.
        If bInterval is present on a INT or ISO endpoint, it must be in
        millisecond units (but may not be an integer), and will be converted
        to the nearest integer millisecond for full-speed descriptor, and
        nearest possible interval for high- and super-speed descriptors.
        If bInterval is present on a BULK endpoint, it is set to zero on
        full-speed descriptor and used as provided on high- and super-speed
        descriptors.
      - "superspeed": optional, contains keyword arguments for
          getDescriptor(USBSSEPCompDescriptor, ...)
      - "superspeed_iso": optional, contains keyword arguments for
          getDescriptor(USBSSPIsocEndpointDescriptor, ...)
        Must be provided and non-empty only when endpoint is isochronous and
        "superspeed" dict has "bmAttributes" bit 7 set.
    class_descriptor (list of descriptors of any type)
      Descriptors to insert in all speeds between the interface descriptor and
      endpoint descriptors.

    Returns a 3-tuple of lists:
    - fs descriptors
    - hs descriptors
    - ss descriptors
    """
    interface = getDescriptor(
        USBInterfaceDescriptor,
        bNumEndpoints=len(endpoint_list),
        **interface
    )
    class_descriptor_list = list(class_descriptor_list)
    fs_list = [interface] + class_descriptor_list
    hs_list = [interface] + class_descriptor_list
    ss_list = [interface] + class_descriptor_list
    need_address = (
        endpoint_list[0]['endpoint'].get(
            'bEndpointAddress',
            0,
        ) & ~ch9.USB_DIR_IN == 0
    )
    for index, endpoint in enumerate(endpoint_list, 1):
        endpoint_kw = endpoint['endpoint'].copy()
        transfer_type = endpoint_kw[
            'bmAttributes'
        ] & ch9.USB_ENDPOINT_XFERTYPE_MASK
        fs_max, hs_max, ss_max = _MAX_PACKET_SIZE_DICT[transfer_type]
        if need_address:
            endpoint_kw['bEndpointAddress'] = index | (
                endpoint_kw.get('bEndpointAddress', 0) & ch9.USB_DIR_IN
            )
        klass = (
            USBEndpointDescriptor
            if 'bRefresh' in endpoint_kw or 'bSynchAddress' in endpoint_kw else
            USBEndpointDescriptorNoAudio
        )
        interval = endpoint_kw.pop('bInterval', _MARKER)
        if interval is _MARKER:
            fs_interval = hs_interval = 0
        else:
            if transfer_type == ch9.USB_ENDPOINT_XFER_BULK:
                fs_interval = 0
                hs_interval = interval
            else: # USB_ENDPOINT_XFER_ISOC or USB_ENDPOINT_XFER_INT
                fs_interval = max(1, min(255, round(interval)))
                # 8 is the number of microframes in a millisecond
                hs_interval = max(
                    1,
                    min(16, int(round(1 + math.log(interval * 8, 2)))),
                )
        packet_size = endpoint_kw.pop('wMaxPacketSize', _MARKER)
        if packet_size is _MARKER:
            fs_packet_size = fs_max
            hs_packet_size = hs_max
            ss_packet_size = ss_max
        else:
            fs_packet_size = min(fs_max, packet_size)
            hs_packet_size = min(hs_max, packet_size)
            ss_packet_size = min(ss_max, packet_size)
        fs_list.append(getDescriptor(
            klass,
            wMaxPacketSize=fs_max,
            bInterval=fs_interval,
            **endpoint_kw
        ))
        hs_list.append(getDescriptor(
            klass,
            wMaxPacketSize=hs_max,
            bInterval=hs_interval,
            **endpoint_kw
        ))
        ss_list.append(getDescriptor(
            klass,
            wMaxPacketSize=ss_max,
            bInterval=hs_interval,
            **endpoint_kw
        ))
        ss_companion_kw = endpoint.get('superspeed', _EMPTY_DICT)
        ss_list.append(getDescriptor(
            USBSSEPCompDescriptor,
            **ss_companion_kw
        ))
        ssp_iso_kw = endpoint.get('superspeed_iso', _EMPTY_DICT)
        if bool(ssp_iso_kw) != (
            endpoint_kw.get('bmAttributes', 0) &
            ch9.USB_ENDPOINT_XFERTYPE_MASK ==
            ch9.USB_ENDPOINT_XFER_ISOC and
            bool(ch9.USB_SS_SSP_ISOC_COMP(
                ss_companion_kw.get('bmAttributes', 0),
            ))
        ):
            raise ValueError('Inconsistent isochronous companion')
        if ssp_iso_kw:
            ss_list.append(getDescriptor(
                USBSSPIsocEndpointDescriptor,
                **ssp_iso_kw
            ))
    return (fs_list, hs_list, ss_list)