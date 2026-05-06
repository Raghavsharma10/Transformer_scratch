def write_multiple_files(args, base_dir, crawler):
    """Write to multiple output files and/or subdirectories."""
    for i, query in enumerate(args['query']):
        if query in args['files']:
            # Write files
            if args['out'] and i < len(args['out']):
                outfilename = args['out'][i]
            else:
                outfilename = '.'.join(query.split('.')[:-1])
            write_files(args, [query], outfilename)
        elif query in args['urls']:
            # Scrape/crawl urls
            domain = utils.get_domain(query)
            if args['html']:
                # Create a directory to save PART.html files in
                if not args['quiet']:
                    print('Storing html files in {0}/'.format(domain))
                utils.mkdir_and_cd(domain)

            if args['crawl'] or args['crawl_all']:
                # Crawl and save HTML files/image files to disk
                infilenames = crawler.crawl_links(query)
            else:
                raw_resp = utils.get_raw_resp(query)
                if raw_resp is None:
                    return False

                # Saves page as PART.html file
                prev_part_num = utils.get_num_part_files()
                utils.write_part_file(args, query, raw_resp)
                curr_part_num = prev_part_num + 1
                infilenames = utils.get_part_filenames(curr_part_num, prev_part_num)

            # Convert output or leave as PART.html files
            if args['html']:
                # HTML files have been written already, so return to base dir
                os.chdir(base_dir)
            else:
                # Write files to text or pdf
                if infilenames:
                    if args['out'] and i < len(args['out']):
                        outfilename = args['out'][i]
                    else:
                        outfilename = utils.get_outfilename(query, domain)
                    write_files(args, infilenames, outfilename)
                else:
                    sys.stderr.write('Failed to retrieve content from {0}.\n'
                                     .format(query))
    return True