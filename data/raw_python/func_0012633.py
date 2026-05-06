def DirectoryStimuliFactory(loader):
    """
    Takes an input path to the images folder of an experiment and generates
    automatically the category - filenumber list needed to construct an
    appropriate _categories object.
    
    Parameters :
        loader : Loader object which contains 
            impath : string
                path to the input, i.e. image-, files of the experiment. All
                subfolders in that path will be treated as categories. If no
                subfolders are present, category 1 will be assigned and all 
                files in the folder are considered input images. 
                Images have to end in '.png'.
            ftrpath : string
                path to the feature folder. It is expected that the folder
                structure corresponds to the structure in impath, i.e. 
                    ftrpath/category/featurefolder/featuremap.mat
                Furthermore, features are assumed to be the same for all 
                categories.
    """
    impath = loader.impath
    ftrpath = loader.ftrpath
    # checks whether user has reading permission for the path
    assert os.access(impath, os.R_OK)
    assert os.access(ftrpath, os.R_OK)    

    # EXTRACTING IMAGE NAMES
    img_per_cat = {}
    # extract only directories in the given folder
    subfolders = [name for name in os.listdir(impath) if os.path.isdir(
        os.path.join(impath, name))]
    # if there are no subfolders, walk through files. Take 1 as key for the 
    # categories object
    if not subfolders:
        [_, _, files] = next(os.walk(os.path.join(impath)))
        # this only takes entries that end with '.png'
        entries = {1: 
            [int(cur_file[cur_file.find('_')+1:-4]) for cur_file
            in files if cur_file.endswith('.png')]}
        img_per_cat.update(entries)
        subfolders = ['']
    # if there are subfolders, walk through them
    else:
        for directory in subfolders:
            [_, _, files] = next(os.walk(os.path.join(impath, directory)))
            # this only takes entries that end with '.png'. Strips ending and
            # considers everything after the first '_' as the imagenumber
            imagenumbers = [int(cur_file[cur_file.find('_')+1:-4]) 
                    for cur_file in files
                        if (cur_file.endswith('.png') & (len(cur_file) > 4))]
            entries = {int(directory): imagenumbers}
            img_per_cat.update(entries)
            del directory
    del imagenumbers

    # in case subfolders do not exist, '' is appended here.
    _, features, files = next(os.walk(os.path.join(ftrpath, 
                                            subfolders[0])))
    return Categories(loader, img_per_cat = img_per_cat, features = features)