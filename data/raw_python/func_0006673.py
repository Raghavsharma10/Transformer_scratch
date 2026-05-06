def write_report(summary_dict, seqid, genus, key):
        """
        Parse the PointFinder outputs, and write the summary report for the current analysis type
        :param summary_dict: nested dictionary containing data such as header strings, and paths to reports
        :param seqid: name of the strain,
        :param genus: MASH-calculated genus of current isolate
        :param key: current result type. Options are 'prediction', and 'results'
        """
        # Set the header string if the summary report doesn't already exist
        if not os.path.isfile(summary_dict[genus][key]['summary']):
            header_string = summary_dict[genus][key]['header']
        else:
            header_string = str()
        summary_string = str()
        try:
            # Read in the predictions
            with open(summary_dict[genus][key]['output'], 'r') as outputs:
                # Skip the header
                next(outputs)
                for line in outputs:
                    # Skip empty lines
                    if line != '\n':
                        # When processing the results outputs, add the seqid to the summary string
                        if key == 'results':
                            summary_string += '{seq},{genus},'.format(seq=seqid,
                                                                      genus=genus)
                        # Clean up the string before adding it to the summary string - replace commas
                        # with semi-colons, and replace tabs with commas
                        summary_string += line.replace(',', ';').replace('\t', ',')
            # Ensure that there were results to report
            if summary_string:
                if not summary_string.endswith('\n'):
                    summary_string += '\n'
            else:
                if key == 'results':
                    summary_string += '{seq},{genus}\n'.format(seq=seqid,
                                                               genus=genus)
                else:
                    summary_string += '{seq}\n'.format(seq=seqid)
            # Write the summaries to the summary file
            with open(summary_dict[genus][key]['summary'], 'a+') as summary:
                # Write the header if necessary
                if header_string:
                    summary.write(header_string)
                summary.write(summary_string)
        # Add the strain information If no FASTA file could be created by reference mapping
        except FileNotFoundError:
            # Extract the length of the header from the dictionary. Subtract two (don't need the strain, or the
            # empty column created by a trailing comma
            header_len = len(summary_dict[genus][key]['header'].split(',')) - 2
            # When processing the results outputs, add the seqid to the summary string
            if key == 'results':
                summary_string += '{seq},{genus}\n'.format(seq=seqid,
                                                           genus=genus)
            # For the prediction summary, populate the summary string with the appropriate number of comma-separated
            # '0' entries
            elif key == 'prediction':
                summary_string += '{seq}{empty}\n'.format(seq=seqid,
                                                          empty=',0' * header_len)
            # Write the summaries to the summary file
            with open(summary_dict[genus][key]['summary'], 'a+') as summary:
                # Write the header if necessary
                if header_string:
                    summary.write(header_string)
                summary.write(summary_string)