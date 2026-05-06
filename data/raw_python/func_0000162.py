def write_addons_items(xml_tree, records, app_id, api_ver=3, app_ver=None):
    """Generate the addons blocklists.

    <emItem blockID="i372" id="5nc3QHFgcb@r06Ws9gvNNVRfH.com">
      <versionRange minVersion="0" maxVersion="*" severity="3">
        <targetApplication id="{ec8030f7-c20a-464f-9b0e-13a3a9e97384}">
          <versionRange minVersion="39.0a1" maxVersion="*"/>
        </targetApplication>
      </versionRange>
      <prefs>
        <pref>browser.startup.homepage</pref>
        <pref>browser.search.defaultenginename</pref>
      </prefs>
    </emItem>
    """
    if not records:
        return

    emItems = etree.SubElement(xml_tree, 'emItems')
    groupby = {}
    for item in records:
        if is_related_to(item, app_id, app_ver):
            if item['guid'] in groupby:
                emItem = groupby[item['guid']]
                # When creating new records from the Kinto Admin we don't have proper blockID.
                if 'blockID' in item:
                    # Remove the first caracter which is the letter i to
                    # compare the numeric value i45 < i356.
                    current_blockID = int(item['blockID'][1:])
                    previous_blockID = int(emItem.attrib['blockID'][1:])
                    # Group by and keep the biggest blockID in the XML file.
                    if current_blockID > previous_blockID:
                        emItem.attrib['blockID'] = item['blockID']
                else:
                    # If the latest entry does not have any blockID attribute, its
                    # ID should be used. (the list of records is sorted by ascending
                    # last_modified).
                    # See https://bugzilla.mozilla.org/show_bug.cgi?id=1473194
                    emItem.attrib['blockID'] = item['id']
            else:
                emItem = etree.SubElement(emItems, 'emItem',
                                          blockID=item.get('blockID', item['id']))
                groupby[item['guid']] = emItem
                prefs = etree.SubElement(emItem, 'prefs')
                for p in item['prefs']:
                    pref = etree.SubElement(prefs, 'pref')
                    pref.text = p

            # Set the add-on ID
            emItem.set('id', item['guid'])

            for field in ['name', 'os']:
                if field in item:
                    emItem.set(field, item[field])

            build_version_range(emItem, item, app_id)