def generate_pos_tagger(check_accuracy=False):
    """Accuracy is about 0.94 with 90% training data."""
    global tagger
    logging.debug("Reading TIGER corpus")
    corp = nltk.corpus.ConllCorpusReader(DIR_PATH, TIGER_FILE_NAME,
                                         ['ignore', 'words', 'ignore', 'ignore', 'pos'],
                                         encoding='utf-8')
    tagged_sents = list(corp.tagged_sents())
    logging.debug("Shuffling sentences")
    random.shuffle(tagged_sents)
    if check_accuracy:
        # set a split size: use 90% for training, 10% for testing
        split_perc = 0.1
        split_size = int(len(tagged_sents) * split_perc)
        train_sents, test_sents = tagged_sents[split_size:], tagged_sents[:split_size]
    else:
        train_sents = tagged_sents
    logging.debug("Training Tagger")
    tagger = ClassifierBasedGermanTagger(train=train_sents)
    logging.debug("Training finished")
    if check_accuracy:
        accuracy = tagger.evaluate(test_sents)
        logging.debug("Accurracy is {}.".format(accuracy))
    logging.debug("Serializing the Tagger")
    with open(os.path.join(DIR_PATH, TAGGER_FILE_NAME), 'wb') as f:
        pickle.dump(tagger, f, protocol=3)