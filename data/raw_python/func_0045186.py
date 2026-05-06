def main():
    """Get a tracklisting, write to audio file or text."""
    args = parse_arguments()
    pid = args.pid
    title = get_programme_title(pid)
    broadcast_date = get_broadcast_date(pid)
    listing = extract_listing(pid)
    filename = get_output_filename(args)
    tracklisting = generate_output(listing, title, broadcast_date)
    output_to_file(filename, tracklisting, args.action)
    print("Done!")