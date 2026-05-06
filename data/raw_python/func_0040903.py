def suffix(filename,suffix):
    ''' returns a filenames with ``suffix`` inserted before the dataset suffix '''
    return os.path.split(re.sub(_afni_suffix_regex,"%s\g<1>" % suffix,str(filename)))[1]