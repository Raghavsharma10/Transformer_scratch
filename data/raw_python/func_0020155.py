def Search(ID, mission='k2'):
    """Why is my target not in the EVEREST database?"""
    # Only K2 supported for now
    assert mission == 'k2', "Only the K2 mission is supported for now."
    print("Searching for target %d..." % ID)

    # First check if it is in the database
    season = missions.k2.Season(ID)
    if season in [91, 92, [91, 92]]:
        print("Campaign 9 is currently not part of the EVEREST catalog.")
        return
    elif season == 101:
        print("The first half of campaign 10 is not currently part of " +
              "the EVEREST catalog.")
        return
    elif season is not None:
        print("Target is in campaign %d of the EVEREST catalog." % season)
        return

    # Get the kplr object
    star = k2plr_client.k2_star(ID)

    # First check if this is a star
    if star.objtype.lower() != "star":
        print("Target is of type %s, not STAR, " % star.objtype +
              "and is therefore not included in the EVEREST catalog.")
        return

    # Let's try to download the pixel data and see what happens
    try:
        tpf = star.get_target_pixel_files()
    except:
        print("Unable to download the raw pixel files for this target.")
        return
    if len(tpf) == 0:
        print("Raw pixel files are not available for this target. Looks like " +
              "data may not have been collected for it.")
        return

    # Perhaps it's in a campaign we haven't gotten to yet
    if tpf[0].sci_campaign not in missions.k2.SEASONS:
        print("Targets for campaign %d are not yet available."
              % tpf[0].sci_campaign)
        return

    # Let's try to download the K2SFF data
    try:
        k2sff = k2plr.K2SFF(ID)
    except:
        print("Error downloading the K2SFF light curve for this target. " +
              "Currently, EVEREST uses the K2SFF apertures to perform " +
              "photometry. This is likely to change in the next version.")
        return

    # Let's try to get the aperture
    try:
        assert np.count_nonzero(k2sff.apertures[15]), "Invalid aperture."
    except:
        print("Unable to retrieve the K2SFF aperture for this target. " +
              "Currently, EVEREST uses the K2SFF apertures to perform " +
              "photometry. This is likely to change in the next version.")
        return

    # Perhaps the star is *super* saturated and we didn't bother
    # de-trending it?
    if star.kp < 8:
        print("Target has Kp = %.1f and is too saturated " +
              "for proper de-trending with EVEREST.")
        return

    # I'm out of ideas
    print("I'm not sure why this target isn't in the EVEREST catalog." +
          "You can try de-trending it yourself:")
    print("http://faculty.washington.edu/rodluger/everest/pipeline.html")
    return