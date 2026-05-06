def kitchen_merge(backend, source_kitchen, target_kitchen):
    """
    Merge two Kitchens
    """
    click.secho('%s - Merging Kitchen %s into Kitchen %s' % (get_datetime(), source_kitchen, target_kitchen), fg='green')
    check_and_print(DKCloudCommandRunner.merge_kitchens_improved(backend.dki, source_kitchen, target_kitchen))