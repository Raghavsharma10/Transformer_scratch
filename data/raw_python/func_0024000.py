def sponsor_menu(
        root_menu, menu="sponsors", label=_("Sponsors"),
        sponsors_item=_("Our sponsors"),
        packages_item=_("Sponsorship packages")):
    """Add sponsor menu links."""
    root_menu.add_menu(menu, label, items=[])
    for sponsor in (
            Sponsor.objects.all()
            .order_by('packages', 'order', 'id')
            .prefetch_related('packages')):
        symbols = sponsor.symbols()
        if symbols:
            item_name = u"» %s %s" % (sponsor.name, symbols)
        else:
            item_name = u"» %s" % (sponsor.name,)
        with menu_logger(logger, "sponsor %r" % (sponsor.name,)):
            root_menu.add_item(
                item_name, sponsor.get_absolute_url(), menu=menu)

    if sponsors_item:
        with menu_logger(logger, "sponsors page link"):
            root_menu.add_item(
                sponsors_item, reverse("wafer_sponsors"), menu)
    if packages_item:
        with menu_logger(logger, "sponsorship package page link"):
            root_menu.add_item(
                packages_item, reverse("wafer_sponsorship_packages"), menu)