def calc_resize_factor(prediction, image_size):
    """
    Calculates how much prediction.shape and image_size differ.
    """
    resize_factor_x = prediction.shape[1] / float(image_size[1])
    resize_factor_y = prediction.shape[0] / float(image_size[0])
    if abs(resize_factor_x - resize_factor_y) > 1.0/image_size[1] :
        raise RuntimeError("""The aspect ratio of the fixations does not
                              match with the prediction: %f vs. %f"""
                              %(resize_factor_y, resize_factor_x))
    return (resize_factor_y, resize_factor_x)