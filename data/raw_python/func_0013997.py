def from_os(cls, data_path=None, format_str=None, 
                two_digit_year_break=None):
        """
        Produces a list of files and and formats it for Files class.

        Requires fixed_width filename
        
        Parameters
        ----------
        data_path : string
            Top level directory to search files for. This directory
            is provided by pysat to the instrument_module.list_files
            functions as data_path.
        format_str : string with python format codes
            Provides the naming pattern of the instrument files and the 
            locations of date information so an ordered list may be produced.
            Supports 'year', 'month', 'day', 'hour', 'min', 'sec', 'version',
            and 'revision'
            Ex: 'cnofs_cindi_ivm_500ms_{year:4d}{month:02d}{day:02d}_v01.cdf'
        two_digit_year_break : int
            If filenames only store two digits for the year, then
            '1900' will be added for years >= two_digit_year_break
            and '2000' will be added for years < two_digit_year_break.
          
        Note
        ----
        Does not produce a Files instance, but the proper output
        from instrument_module.list_files method.

        The '?' may be used to indicate a set number of spaces for a variable
        part of the name that need not be extracted.
        'cnofs_cindi_ivm_500ms_{year:4d}{month:02d}{day:02d}_v??.cdf'
        """

        import collections
        
        from pysat.utils import create_datetime_index
        
        if format_str is None:
            raise ValueError("Must supply a filename template (format_str).")
        if data_path is None:
            raise ValueError("Must supply instrument directory path (dir_path)")
        
        # parse format string to figure out the search string to use
        # to identify files in the filesystem
        search_str = ''
        form = string.Formatter()
        # stores the keywords extracted from format_string
        keys = []
        #, and length of string
        snips = []
        length = []
        stored = collections.OrderedDict()
        stored['year'] = []; stored['month'] = []; stored['day'] = [];
        stored['hour'] = []; stored['min'] = []; stored['sec'] = [];
        stored['version'] = []; stored['revision'] = [];
        for snip in form.parse(format_str):
            # collect all of the format keywords
            # replace them in the string with the '*' wildcard
            # then try and get width from format keywords so we know
            # later on where to parse information out from
            search_str += snip[0]
            snips.append(snip[0])
            if snip[1] is not None:
                keys.append(snip[1])
                search_str += '*'
                # try and determine formatting width
                temp = re.findall(r'\d+', snip[2])
                if temp:
                    # there are items, try and grab width
                    for i in temp:
                        if i != 0:
                            length.append(int(i))
                            break
                else:
                    raise ValueError("Couldn't determine formatting width")

        abs_search_str = os.path.join(data_path, search_str)
        files = glob.glob(abs_search_str)   
        
        # we have a list of files, now we need to extract the date information
        # code below works, but only if the size of file string 
        # remains unchanged
        
        # determine the locations the date information in a filename is stored
        # use these indices to slice out date from loaded filenames
        # test_str = format_str.format(**periods)
        if len(files) > 0:  
            idx = 0
            begin_key = []
            end_key = []
            for i,snip in enumerate(snips):
                idx += len(snip)
                if i < (len(length)):
                    begin_key.append(idx)
                    idx += length[i]
                    end_key.append(idx)
            max_len = idx
            # setting up negative indexing to pick out filenames
            key_str_idx = [np.array(begin_key, dtype=int) - max_len, 
                           np.array(end_key, dtype=int) - max_len]
            # need to parse out dates for datetime index
            for i,temp in enumerate(files):
                for j,key in enumerate(keys):
                    val = temp[key_str_idx[0][j]:key_str_idx[1][j]]
                    stored[key].append(val)
            # convert to numpy arrays
            for key in stored.keys():
                stored[key] = np.array(stored[key]).astype(int)
                if len(stored[key]) == 0:
                    stored[key]=None
            # deal with the possibility of two digit years
            # years above or equal to break are considered to be 1900+
            # years below break are considered to be 2000+
            if two_digit_year_break is not None:
                idx, = np.where(np.array(stored['year']) >=
                                two_digit_year_break)
                stored['year'][idx] = stored['year'][idx] + 1900
                idx, = np.where(np.array(stored['year']) < two_digit_year_break)
                stored['year'][idx] = stored['year'][idx] + 2000 
            # need to sort the information for things to work
            rec_arr = [stored[key] for key in keys]
            rec_arr.append(files)
            # sort all arrays
            val_keys = keys + ['files']
            rec_arr = np.rec.fromarrays(rec_arr, names=val_keys)
            rec_arr.sort(order=val_keys, axis=0)
            # pull out sorted info
            for key in keys:
                stored[key] = rec_arr[key]
            files = rec_arr['files']
            # add hour and minute information to 'sec'
            if stored['sec'] is None:
                stored['sec'] = np.zeros(len(files))                
            if stored['hour'] is not None:
                stored['sec'] += 3600 * stored['hour']
            if stored['min'] is not None:
                stored['sec'] += 60 * stored['min']
            # if stored['version'] is None:
            #     stored['version'] = np.zeros(len(files))
            if stored['revision'] is None:
                stored['revision'] = np.zeros(len(files))

            index = create_datetime_index(year=stored['year'],
                                          month=stored['month'], 
                                          day=stored['day'], uts=stored['sec'])

            # if version and revision are supplied
            # use these parameters to weed out files that have been replaced
            # with updated versions
            # first, check for duplicate index times
            dups = index.get_duplicates()
            if (len(dups) > 0) and (stored['version'] is not None):
                # we have duplicates
                # keep the highest version/revision combo
                version = pds.Series(stored['version'], index=index)
                revision = pds.Series(stored['revision'], index=index)
                revive = version*100000. + revision
                frame = pds.DataFrame({'files':files, 'revive':revive,
                                       'time':index}, index=index)
                frame = frame.sort_values(by=['time', 'revive'],
                                          ascending=[True, False])
                frame = frame.drop_duplicates(subset='time', keep='first')

                return frame['files']
            else:
                return pds.Series(files, index=index)
        else:
            return pds.Series(None)