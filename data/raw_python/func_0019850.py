def load_example_data(dataset='automatic'):
    """
    Loads example data

    The automatic and manual example data are eggs containing 30 subjects who completed a free
    recall experiment as described here: https://psyarxiv.com/psh48/. The subjects
    studied 8 lists of 16 words each and then performed a free recall test.

    The naturalistic example data is is an egg containing 17 subjects who viewed and verbally
    recounted an episode of the BBC series Sherlock, as described here:
    https://www.nature.com/articles/nn.4450. We fit a topic model to hand-annotated
    text-descriptions of scenes from the video and used the model to transform both the
    scene descriptions and manual transcriptions of each subject's verbal recall. We then
    used a Hidden Markov Model to segment the video model and the recall models, by subject,
    into k events.

    Parameters
    ----------
    dataset : str
        The dataset to load. Can be 'automatic', 'manual', or 'naturalistic'. The free recall
        audio recordings for the 'automatic' dataset was transcribed by Google
        Cloud Speech and the 'manual' dataset was transcribed by humans. The 'naturalistic'
        dataset was transcribed by humans and transformed as described above.

    Returns
    ----------
    data : quail.Egg
        Example data
    """

    # can only be auto or manual
    assert dataset in ['automatic', 'manual', 'naturalistic'], "Dataset can only be automatic, manual, or naturalistic"


    if dataset == 'naturalistic':
        # open naturalistic egg
        egg = Egg(**dd.io.load(os.path.dirname(os.path.abspath(__file__)) + '/data/' + dataset + '.egg'))
    else:
    # open pickled egg
        try:
            with open(os.path.dirname(os.path.abspath(__file__)) + '/data/' + dataset + '.egg', 'rb') as handle:
                egg = pickle.load(handle)
        except:
            f = dd.io.load(os.path.dirname(os.path.abspath(__file__)) + '/data/' + dataset + '.egg')
            egg = Egg(pres=f['pres'], rec=f['rec'], dist_funcs=f['dist_funcs'],
                      subjgroup=f['subjgroup'], subjname=f['subjname'],
                      listgroup=f['listgroup'], listname=f['listname'],
                      date_created=f['date_created'])
    return egg.crack()