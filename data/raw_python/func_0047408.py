def infer(self, number_of_processes=1, *args, **kwargs):
        """
        :param number_of_processes: If set to more than 1, the inference routines will be paralellised
                                    using ``multiprocessing`` module
        :param args: arguments to pass to :meth:`Inference.infer`
        :param kwargs: keyword arguments to pass to :meth:`Inference.infer`
        :return:
        """
        if number_of_processes == 1:
            results = map(lambda x: x.infer(*args, **kwargs), self._inference_objects)
        else:
            inference_objects = self._inference_objects
            results = raw_results_in_parallel(self._inference_objects, number_of_processes, *args,
                                              **kwargs)
            results = [inference._result_from_raw_result(raw_result)
                       for inference, raw_result in zip(inference_objects, results)]


        results = sorted(results, key=lambda x: x.distance_at_minimum)

        return InferenceResultsCollection(results)