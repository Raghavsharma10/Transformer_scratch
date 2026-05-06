def multiprocessing_apply_infer(object_id):
    """
    Used in the InferenceWithRestarts class.
    Needs to be in global scope for multiprocessing module to pick it up

    """
    global inference_objects, inference_args, inference_kwargs
    return inference_objects[object_id]._infer_raw(*inference_args, **inference_kwargs)