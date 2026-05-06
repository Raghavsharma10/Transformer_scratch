def write_gfx_items(xml_tree, records, app_id, api_ver=3):
    """Generate the gfxBlacklistEntry.

    <gfxBlacklistEntry blockID="g35">
        <os>WINNT 6.1</os>
        <vendor>0x10de</vendor>
        <devices>
            <device>0x0a6c</device>
        </devices>
        <feature>DIRECT2D</feature>
        <featureStatus>BLOCKED_DRIVER_VERSION</featureStatus>
        <driverVersion>8.17.12.5896</driverVersion>
        <driverVersionComparator>LESS_THAN_OR_EQUAL</driverVersionComparator>
        <versionRange minVersion="3.2" maxVersion="3.4" />
    </gfxBlacklistEntry>
    """
    if not records:
        return

    gfxItems = etree.SubElement(xml_tree, 'gfxItems')
    for item in records:
        is_record_related = ('guid' not in item or item['guid'] == app_id)

        if is_record_related:
            entry = etree.SubElement(gfxItems, 'gfxBlacklistEntry',
                                     blockID=item.get('blockID', item['id']))
            fields = ['os', 'vendor', 'feature', 'featureStatus',
                      'driverVersion', 'driverVersionComparator']
            for field in fields:
                if field in item:
                    node = etree.SubElement(entry, field)
                    node.text = item[field]

            # Devices
            if item['devices']:
                devices = etree.SubElement(entry, 'devices')
                for d in item['devices']:
                    device = etree.SubElement(devices, 'device')
                    device.text = d

            if 'versionRange' in item:
                version = item['versionRange']
                versionRange = etree.SubElement(entry, 'versionRange')

                for field in ['minVersion', 'maxVersion']:
                    value = version.get(field)
                    if value:
                        versionRange.set(field, str(value))