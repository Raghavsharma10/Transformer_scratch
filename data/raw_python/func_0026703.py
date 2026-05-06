def doblob(morphed, blobdet, img, anno=True):
    """
    img: can be RGB (MxNx3) or gray (MxN)
    http://docs.opencv.org/master/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
    http://docs.opencv.org/trunk/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
    """
    keypoints = blobdet.detect(morphed)
    nkey = len(keypoints)
    kpsize = asarray([k.size for k in keypoints])
    final = img.copy()  # is the .copy necessary?

    final = cv2.drawKeypoints(img, keypoints, outImage=final,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# %% plot count of blobs
    if anno:
        cv2.putText(final, text=str(nkey), org=(int(img.shape[1]*.9), 25),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                    color=(0, 255, 0), thickness=2)

    return final, nkey, kpsize