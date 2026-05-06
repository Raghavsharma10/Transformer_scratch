def tagsInString_process(self, d_DICOM, astr, *args, **kwargs):
        """
        This method substitutes DICOM tags that are '%'-tagged
        in a string template with the actual tag lookup.

        For example, an output filename that is specified as the
        following string:

            %PatientAge-%PatientID-output.txt

        will be parsed to

            006Y-4412364-ouptut.txt

        It is also possible to apply certain permutations/functions
        to a tag. For example, a function is identified by an underscore
        prefixed and suffixed string as part of the DICOM tag. If
        found, this function is applied to the tag value. For example,

            %PatientAge-%_md5|4_PatientID-output.txt

        will apply an md5 hash to the PatientID and use the first 4
        characters:

            006Y-7f38-output.txt

        """
        b_tagsFound         = False
        str_replace         = ''        # The lookup/processed tag value
        l_tags              = []        # The input string split by '%'
        l_tagsToSub         = []        # Remove any noise etc from each tag
        l_funcTag           = []        # a function/tag list
        l_args              = []        # the 'args' of the function
        func                = ''        # the function to apply
        tag                 = ''        # the tag in the funcTag combo
        chars               = ''        # the number of resultant chars from func
                                        # result to use
        if '%' in astr:
            l_tags          = astr.split('%')[1:]
            # Find which tags (mangled) in string match actual tags
            l_tagsToSub     = [i for i in d_DICOM['l_tagRaw'] if any(i in b for b in l_tags)]
            # Need to arrange l_tagsToSub in same order as l_tags
            l_tagsToSubSort =  sorted(
                l_tagsToSub, 
                key = lambda x: [i for i, s in enumerate(l_tags) if x in s][0]
            )
            for tag, func in zip(l_tagsToSubSort, l_tags):
                b_tagsFound     = True
                str_replace     = d_DICOM['d_dicomSimple'][tag]
                if 'md5' in func:
                    str_replace = hashlib.md5(str_replace.encode('utf-8')).hexdigest()
                    l_funcTag   = func.split('_')[1:]
                    func        = l_funcTag[0]
                    l_args      = func.split('|')
                    if len(l_args) > 1:
                        chars   = l_args[1]
                        str_replace     = str_replace[0:int(chars)]
                    astr = astr.replace('_%s_' % func, '')
                if 'strmsk' in func:
                    l_funcTag   = func.split('_')[1:]
                    func        = l_funcTag[0]
                    str_msk     = func.split('|')[1]
                    l_n = []
                    for i, j in zip(list(str_replace), list(str_msk)):
                        if j == '*':    l_n.append(i)
                        else:           l_n.append(j)
                    str_replace = ''.join(l_n)
                    astr = astr.replace('_%s_' % func, '')
                if 'nospc' in func:
                    # pudb.set_trace()
                    l_funcTag   = func.split('_')[1:]
                    func        = l_funcTag[0]
                    l_args      = func.split('|')
                    str_char    = ''
                    if len(l_args) > 1:
                        str_char = l_args[1]
                    # strip out all non-alphnumeric chars and 
                    # replace with space
                    str_replace = re.sub(r'\W+', ' ', str_replace)
                    # replace all spaces with str_char
                    str_replace = str_char.join(str_replace.split())
                    astr = astr.replace('_%s_' % func, '')
                astr  = astr.replace('%' + tag, str_replace)
        
        return {
            'status':       True,
            'b_tagsFound':  b_tagsFound,
            'str_result':   astr
        }