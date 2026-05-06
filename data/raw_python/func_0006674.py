def write_table_report(summary_dict, seqid, genus):
        """
        Parse the PointFinder table output, and write a summary report
        :param summary_dict: nested dictionary containing data such as header strings, and paths to reports
        :param seqid: name of the strain,
        :param genus: MASH-calculated genus of current isolate
        """
        # Set the header string if the summary report doesn't already exist
        if not os.path.isfile(summary_dict[genus]['table']['summary']):
            header_string = summary_dict[genus]['table']['header']
        else:
            header_string = str()
        summary_string = '{seq},'.format(seq=seqid)
        try:
            # Read in the predictions
            with open(summary_dict[genus]['table']['output'], 'r') as outputs:
                for header_value in summary_dict[genus]['table']['header'].split(',')[:-1]:
                    for line in outputs:
                        if line.startswith('{hv}\n'.format(hv=header_value)):
                            # Iterate through the lines following the match
                            for subline in outputs:
                                if subline != '\n':
                                    if subline.startswith('Mutation'):
                                        for detailline in outputs:
                                            if detailline != '\n':
                                                summary_string += '{},'.format(detailline.split('\t')[0])
                                            else:
                                                break
                                    else:
                                        summary_string += '{},'.format(
                                            subline.replace(',', ';').replace('\t', ',').rstrip())
                                        break
                                else:
                                    break
                                break
                    # Reset the file iterator to the first line in preparation for the next header
                    outputs.seek(0)
            # Ensure that there were results to report
            if summary_string:
                if not summary_string.endswith('\n'):
                    summary_string += '\n'
                # Write the summaries to the summary file
                with open(summary_dict[genus]['table']['summary'], 'a+') as summary:
                    # Write the header if necessary
                    if header_string:
                        summary.write(header_string)
                    summary.write(summary_string)
        except FileNotFoundError:
            # Write the summaries to the summary file
            with open(summary_dict[genus]['table']['summary'], 'a+') as summary:
                # Extract the length of the header from the dictionary. Subtract two (don't need the strain, or the
                # empty column created by a trailing comma
                header_len = len(summary_dict[genus]['table']['header'].split(',')) - 2
                # Populate the summary strain with the appropriate number of comma-separated 'Gene not found' entries
                summary_string += '{empty}\n'.format(empty='Gene not found,' * header_len)
                # Write the header if necessary
                if header_string:
                    summary.write(header_string)
                summary.write(summary_string)