def K2findCampaigns_main(args=None):
    """Exposes K2findCampaigns to the command line."""
    parser = argparse.ArgumentParser(
                    description="Check if a celestial coordinate is "
                                "(or was) observable by any past or future "
                                "observing campaign of NASA's K2 mission.")
    parser.add_argument('ra', nargs=1, type=float,
                        help="Right Ascension in decimal degrees (J2000).")
    parser.add_argument('dec', nargs=1, type=float,
                        help="Declination in decimal degrees (J2000).")
    parser.add_argument('-p', '--plot', action='store_true',
                        help="Produce a plot showing the target position "
                             "with respect to all K2 campaigns.")
    args = parser.parse_args(args)
    ra, dec = args.ra[0], args.dec[0]
    campaigns = findCampaigns(ra, dec)
    # Print the result
    if len(campaigns):
        print(Highlight.GREEN + "Success! The target is on silicon "
              "during K2 campaigns {0}.".format(campaigns) + Highlight.END)
    else:
        print(Highlight.RED + "Sorry, the target is not on silicon "
              "during any K2 campaign." + Highlight.END)
    # Print the pixel positions
    for c in campaigns:
        printChannelColRow(c, ra, dec)
    # Make a context plot if the user requested so
    if args.plot:
        save_context_plots(ra, dec, "Your object")