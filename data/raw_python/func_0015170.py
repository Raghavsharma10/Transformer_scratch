def format_daps(simple=False, skip_installed=False):
    '''Formats a list of metadaps available on Dapi'''
    lines= []
    m = metadaps()
    if not m['count']:
        logger.info('Could not find any daps')
        return
    for mdap in sorted(m['results'], key=lambda mdap: mdap['package_name']):
        if skip_installed and mdap['package_name'] in get_installed_daps():
            continue
        if simple:
            logger.info(mdap['package_name'])
        else:
            for line in _format_dap_with_description(mdap):
                lines.append(line)
    return lines