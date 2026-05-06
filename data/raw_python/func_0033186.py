def train_rdp_classifier(
        training_seqs_file, taxonomy_file, model_output_dir, max_memory=None,
        tmp_dir=tempfile.gettempdir()):
    """ Train RDP Classifier, saving to model_output_dir

        training_seqs_file, taxonomy_file: file-like objects used to
            train the RDP Classifier (see RdpTrainer documentation for
            format of training data)

        model_output_dir: directory in which to save the files
            necessary to classify sequences according to the training
            data

    Once the model data has been generated, the RDP Classifier may
    """
    app_kwargs = {}
    if tmp_dir is not None:
        app_kwargs['TmpDir'] = tmp_dir
    app = RdpTrainer(**app_kwargs)

    if max_memory is not None:
        app.Parameters['-Xmx'].on(max_memory)

    temp_taxonomy_file = tempfile.NamedTemporaryFile(
        prefix='RdpTaxonomy_', suffix='.txt', dir=tmp_dir)
    temp_taxonomy_file.write(taxonomy_file.read())
    temp_taxonomy_file.seek(0)

    app.Parameters['taxonomy_file'].on(temp_taxonomy_file.name)
    app.Parameters['model_output_dir'].on(model_output_dir)
    return app(training_seqs_file)