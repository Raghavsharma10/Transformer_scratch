def classify(label_dict,image_fname=None,image_label=None):
    '''tries to classify a DICOM image based on known string patterns (with fuzzy matching)

    Takes the label from the DICOM header and compares to the entries in ``label_dict``. If it finds something close
    it will return the image type, otherwise it will return ``None``. Alternatively, you can supply your own string, ``image_label``,
    and it will try to match that.

    ``label_dict`` is a dictionary where the keys are dataset types and the values are lists of strings that match that type.
    For example::

        {
            'anatomy': ['SPGR','MPRAGE','anat','anatomy'],
            'dti': ['DTI'],
            'field_map': ['fieldmap','TE7','B0']
        }
    '''
    min_acceptable_match = 80
    if image_fname:
        label_info = info_for_tags(image_fname,[(0x8,0x103e)])
        image_label = label_info[(0x8,0x103e)]
    # creates a list of tuples: (type, keyword)
    flat_dict = [i for j in [[(b,x) for x in label_dict[b]]  for b in label_dict] for i in j]
    best_match = process.extractOne(image_label,[x[1] for x in flat_dict])
    if best_match[1]<min_acceptable_match:
        return None
    else:
        return [x[0] for x in flat_dict if x[1]==best_match[0]][0]