def download_from_plugin(plugin: APlugin):
    """
    Download routine.

    1. get newest update time
    2. load savestate
    3. compare last update time with savestate time
    4. get download links
    5. compare with savestate
    6. download new/updated data
    7. check downloads
    8. update savestate
    9. write new savestate

    :param plugin: plugin
    :type plugin: ~unidown.plugin.a_plugin.APlugin
    """
    # get last update date
    plugin.log.info('Get last update')
    plugin.update_last_update()
    # load old save state
    save_state = plugin.load_save_state()
    if plugin.last_update <= save_state.last_update:
        plugin.log.info('No update. Nothing to do.')
        return
    # get download links
    plugin.log.info('Get download links')
    plugin.update_download_links()
    # compare with save state
    down_link_item_dict = plugin.get_updated_data(save_state.link_item_dict)
    plugin.log.info('Compared with save state: ' + str(len(plugin.download_data)))
    if not down_link_item_dict:
        plugin.log.info('No new data. Nothing to do.')
        return
    # download new/updated data
    plugin.log.info(f"Download new {plugin.unit}s: {len(down_link_item_dict)}")
    plugin.download(down_link_item_dict, plugin.download_path, 'Download new ' + plugin.unit + 's', plugin.unit)
    # check which downloads are succeeded
    succeed_link_item_dict, lost_link_item_dict = plugin.check_download(down_link_item_dict, plugin.download_path)
    plugin.log.info(f"Downloaded: {len(succeed_link_item_dict)}/{len(down_link_item_dict)}")
    # update savestate link_item_dict with succeeded downloads dict
    plugin.log.info('Update savestate')
    plugin.update_dict(save_state.link_item_dict, succeed_link_item_dict)
    # write new savestate
    plugin.log.info('Write savestate')
    plugin.save_save_state(save_state.link_item_dict)