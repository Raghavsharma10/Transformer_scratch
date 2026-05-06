def emd(prediction, ground_truth):
    """
    Compute the Eart Movers Distance between prediction and model.

    This implementation uses opencv for doing the actual work.
    Unfortunately, at the time of implementation only the SWIG
    bindings werer available and the numpy arrays have to
    converted by hand. This changes with opencv 2.1.  """
    import opencv

    if not (prediction.shape == ground_truth.shape):
        raise RuntimeError('Shapes of prediction and ground truth have' +
                           ' to be equal. They are: %s, %s'
                            %(str(prediction.shape), str(ground_truth.shape)))
    (x, y) = np.meshgrid(list(range(0, prediction.shape[1])),
                        list(range(0, prediction.shape[0])))
    s1 = np.array([x.flatten(), y.flatten(), prediction.flatten()]).T
    s2 = np.array([x.flatten(), y.flatten(), ground_truth.flatten()]).T
    s1m = opencv.cvCreateMat(s1.shape[0], s2.shape[1], opencv.CV_32FC1)
    s2m = opencv.cvCreateMat(s1.shape[0], s2.shape[1], opencv.CV_32FC1)
    for r in range(0, s1.shape[0]):
        for c in range(0, s1.shape[1]):
            s1m[r, c] = float(s1[r, c])
            s2m[r, c] = float(s2[r, c])
    d = opencv.cvCalcEMD2(s1m, s2m, opencv.CV_DIST_L2)
    return d