def process_error_labels(value):
        """ Process the error labels of a dependent variable 'value' to ensure uniqueness. """

        observed_error_labels = {}
        for error in value.get('errors', []):
            label = error.get('label', 'error')

            if label not in observed_error_labels:
                observed_error_labels[label] = 0
            observed_error_labels[label] += 1

            if observed_error_labels[label] > 1:
                error['label'] = label + '_' + str(observed_error_labels[label])

            # append "_1" to first error label that has a duplicate
            if observed_error_labels[label] == 2:
                for error1 in value.get('errors', []):
                    error1_label = error1.get('label', 'error')
                    if error1_label == label:
                        error1['label'] = label + "_1"
                        break