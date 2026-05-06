def train_rdp_classifier_and_assign_taxonomy(
        training_seqs_file, taxonomy_file, seqs_to_classify, min_confidence=0.80,
        model_output_dir=None, classification_output_fp=None, max_memory=None,
        tmp_dir=tempfile.gettempdir()):
    """ Train RDP Classifier and assign taxonomy in one fell swoop

    The file objects training_seqs_file and taxonomy_file are used to
    train the RDP Classifier (see RdpTrainer documentation for
    details).  Model data is stored in model_output_dir.  If
    model_output_dir is not provided, a temporary directory is created
    and removed after classification.

    The sequences in seqs_to_classify are classified according to the
    model and filtered at the desired confidence level (default:
    0.80).

    The results are saved to classification_output_fp if provided,
    otherwise a dict of {seq_id:(taxonomy_assignment,confidence)} is
    returned.
    """
    if model_output_dir is None:
        training_dir = tempfile.mkdtemp(prefix='RdpTrainer_', dir=tmp_dir)
    else:
        training_dir = model_output_dir

    training_results = train_rdp_classifier(
        training_seqs_file, taxonomy_file, training_dir, max_memory=max_memory,
        tmp_dir=tmp_dir)
    training_data_fp = training_results['properties'].name

    assignment_results = assign_taxonomy(
        seqs_to_classify, min_confidence=min_confidence,
        output_fp=classification_output_fp, training_data_fp=training_data_fp,
        max_memory=max_memory, fixrank=False, tmp_dir=tmp_dir)

    if model_output_dir is None:
        # Forum user reported an error on the call to os.rmtree:
        # https://groups.google.com/d/topic/qiime-forum/MkNe7-JtSBw/discussion
        # We were not able to replicate the problem and fix it
        # properly.  However, even if an error occurs, we would like
        # to return results, along with a warning.
        try:
            rmtree(training_dir)
        except OSError:
            msg = (
                "Temporary training directory %s not removed" % training_dir)
            if os.path.isdir(training_dir):
                training_dir_files = os.listdir(training_dir)
                msg += "\nDetected files %s" % training_dir_files
            warnings.warn(msg, RuntimeWarning)

    return assignment_results