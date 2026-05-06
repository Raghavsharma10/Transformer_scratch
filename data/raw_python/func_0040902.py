def prefix(filename):
    ''' strips common fMRI dataset suffixes from filenames '''
    return os.path.split(re.sub(_afni_suffix_regex,"",str(filename)))[1]