def K2findCampaigns_byname_main(args=None):
    """Exposes K2findCampaigns to the command line."""
    parser = argparse.ArgumentParser(
                    description="Check if a target is "
                                "(or was) observable by any past or future "
                                "observing campaign of NASA's K2 mission.")
    parser.add_argument('name', nargs=1, type=str,
                        help="Name of the object.  This will be passed on "
                             "to the CDS name resolver "
                             "to retrieve coordinate information.")
    parser.add_argument('-p', '--plot', action='store_true',
                        help="Produce a plot showing the target position "
                             "with respect to all K2 campaigns.")
    args = parser.parse_args(args)
    targetname = args.name[0]
    try:
        campaigns, ra, dec = findCampaignsByName(targetname)
    except ValueError:
        print("Error: could not retrieve coordinates for {0}.".format(targetname))
        print("The target may be unknown or there may be a problem "
              "connecting to the coordinate server.")
        sys.exit(1)
    # Print the result
    if len(campaigns):
        print(Highlight.GREEN +
              "Success! {0} is on silicon ".format(targetname) +
              "during K2 campaigns {0}.".format(campaigns) +
              Highlight.END)
    else:
        print(Highlight.RED + "Sorry, {} is not on silicon "
              "during any K2 campaign.".format(targetname) + Highlight.END)
    # Print the pixel positions
    for c in campaigns:
        printChannelColRow(c, ra, dec)
    # Make a context plot if the user requested so
    if args.plot:
        save_context_plots(ra, dec, targetname=targetname)