def K2findCampaigns_csv_main(args=None):
    """Exposes K2findCampaigns-csv to the command line."""
    parser = argparse.ArgumentParser(
                    description="Check which objects listed in a CSV table "
                                "are (or were) observable by NASA's K2 mission.")
    parser.add_argument('input_filename', nargs=1, type=str,
                        help="Path to a comma-separated table containing "
                             "columns 'ra,dec,kepmag' (decimal degrees) "
                             "or 'name'.")
    args = parser.parse_args(args)
    input_fn = args.input_filename[0]
    output_fn = input_fn + '-K2findCampaigns.csv'
    # First, try assuming the file has the classic "ra,dec,kepmag" format
    try:
        ra, dec, kepmag = parse_file(input_fn, exit_on_error=False)
        campaigns = np.array([findCampaigns(ra[idx], dec[idx])
                              for idx in range(len(ra))])
        output = np.array([ra, dec, kepmag, campaigns])
        print("Writing {0}".format(output_fn))
        np.savetxt(output_fn, output.T, delimiter=', ',
                   fmt=['%10.10f', '%10.10f', '%10.2f', '%s'])
    # If this fails, assume the file has a single "name" column
    except ValueError:
        names = [name.strip() for name in open(input_fn, "r").readlines()
                 if len(name.strip()) > 0]
        print("Writing {0}".format(output_fn))
        output = open(output_fn, "w")
        for target in names:
            try:
                campaigns, ra, dec = findCampaignsByName(target)
            except ValueError:
                campaigns = []
            output.write("{0}, {1}\n".format(target, campaigns))
            output.flush()
        output.close()