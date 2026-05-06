def write_plugin_items(xml_tree, records, app_id, api_ver=3, app_ver=None):
    """Generate the plugin blocklists.

    <pluginItem blockID="p422">
        <match name="filename" exp="JavaAppletPlugin\\.plugin"/>
        <versionRange minVersion="Java 7 Update 16"
                      maxVersion="Java 7 Update 24"
                      severity="0" vulnerabilitystatus="1">
            <targetApplication id="{ec8030f7-c20a-464f-9b0e-13a3a9e97384}">
                <versionRange minVersion="17.0" maxVersion="*"/>
            </targetApplication>
        </versionRange>
    </pluginItem>
    """

    if not records:
        return

    pluginItems = etree.SubElement(xml_tree, 'pluginItems')
    for item in records:
        for versionRange in item.get('versionRange', []):
            if not versionRange.get('targetApplication'):
                add_plugin_item(pluginItems, item, versionRange,
                                app_id=app_id, api_ver=api_ver,
                                app_ver=app_ver)
            else:
                targetApplication = get_related_targetApplication(versionRange, app_id, app_ver)
                if targetApplication is not None:
                    add_plugin_item(pluginItems, item, versionRange, targetApplication,
                                    app_id=app_id, api_ver=api_ver,
                                    app_ver=app_ver)