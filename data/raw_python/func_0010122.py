def write_single_file(args, base_dir, crawler):
    """Write to a single output file and/or subdirectory."""
    if args['urls'] and args['html']:
        # Create a directory to save PART.html files in
        domain = utils.get_domain(args['urls'][0])
        if not args['quiet']:
            print('Storing html files in {0}/'.format(domain))
        utils.mkdir_and_cd(domain)

    infilenames = []
    for query in args['query']:
        if query in args['files']:
            infilenames.append(query)
        elif query.strip('/') in args['urls']:
            if args['crawl'] or args['crawl_all']:
                # Crawl and save HTML files/image files to disk
                infilenames += crawler.crawl_links(query)
            else:
                raw_resp = utils.get_raw_resp(query)
                if raw_resp is None:
                    return False

                prev_part_num = utils.get_num_part_files()
                utils.write_part_file(args, query, raw_resp)
                curr_part_num = prev_part_num + 1
                infilenames += utils.get_part_filenames(curr_part_num, prev_part_num)

    # Convert output or leave as PART.html files
    if args['html']:
        # HTML files have been written already, so return to base directory
        os.chdir(base_dir)
    else:
        # Write files to text or pdf
        if infilenames:
            if args['out']:
                outfilename = args['out'][0]
            else:
                outfilename = utils.get_single_outfilename(args)
            if outfilename:
                write_files(args, infilenames, outfilename)
        else:
            utils.remove_part_files()
    return True