def FixmatStimuliFactory(fm, loader):
    """
    Constructs an categories object for all image / category 
    combinations in the fixmat.
    
    Parameters:
        fm: FixMat
            Used for extracting valid category/image combination.
        loader: loader
            Loader that accesses the stimuli for this fixmat
 
    Returns:
        Categories object
    """
    # Find all feature names
    features = [] 
    if loader.ftrpath:
        assert os.access(loader.ftrpath, os.R_OK)   
        features = os.listdir(os.path.join(loader.ftrpath, str(fm.category[0])))
    # Find all images in all categories   
    img_per_cat = {}
    for cat in np.unique(fm.category):
        if not loader.test_for_category(cat):
            raise ValueError('Category %s is specified in fixmat but '%(
                                str(cat) + 'can not be located by loader'))
        img_per_cat[cat] = []
        for img in np.unique(fm[(fm.category == cat)].filenumber):
            if not loader.test_for_image(cat, img):
                raise ValueError('Image %s in category %s is '%(str(cat), 
                    str(img)) + 
                    'specified in fixmat but can be located by loader')
            img_per_cat[cat].append(img)
            if loader.ftrpath:
                for feature in features:
                    if not loader.test_for_feature(cat, img, feature):
                        raise RuntimeError(
                            'Feature %s for image %s' %(str(feature),str(img)) +
                            ' in category %s ' %str(cat) +
                            'can not be located by loader') 
    return Categories(loader, img_per_cat = img_per_cat,
         features = features, fixations = fm)