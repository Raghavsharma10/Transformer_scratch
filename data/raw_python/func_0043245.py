def uninstall(args):
    " Uninstall site. "

    site = find_site(args.PATH)
    site.run_remove()
    site.clean()
    if not listdir(op.dirname(site.deploy_dir)):
        call('sudo rm -rf %s' % op.dirname(site.deploy_dir))