def write_text(filename, tracklisting):
    """Handle writing tracklisting to text."""
    print("Saving text file.")
    try:
        write_listing_to_textfile(filename + '.txt', tracklisting)
    except IOError:
        # if all else fails, just print listing
        print("Cannot write text file to path: {}".format(filename))
        print("Printing tracklisting here instead.")
        # ignoring errors is a hack to cope with Windows not dealing well
        # with UTF-8
        print(tracklisting.encode(sys.stdout.encoding, errors='ignore'))