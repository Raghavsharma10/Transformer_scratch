def __load_arff(self, arff_path, encode_nonnumeric=False):
        """Loads a given dataset saved in Weka's ARFF format. """
        try:
            from scipy.io.arff import loadarff
            arff_data, arff_meta = loadarff(arff_path)
        except:
            raise ValueError('Error loading the ARFF dataset!')

        attr_names = arff_meta.names()[:-1]  # last column is class
        attr_types = arff_meta.types()[:-1]
        if not encode_nonnumeric:
            # ensure all the attributes are numeric
            uniq_types = set(attr_types)
            if 'numeric' not in uniq_types:
                raise ValueError(
                    'Currently only numeric attributes in ARFF are supported!')

            non_numeric = uniq_types.difference({'numeric'})
            if len(non_numeric) > 0:
                raise ValueError('Non-numeric features provided ({}), '
                                 'without requesting encoding to numeric. '
                                 'Try setting encode_nonnumeric=True '
                                 'or encode features to numeric!'.format(non_numeric))
        else:
            raise NotImplementedError(
                'encoding non-numeric features to numeric is not implemented yet! '
                'Encode features beforing to ARFF.')

        self.__description = arff_meta.name  # to enable it as a label e.g. in neuropredict

        # initializing the key containers, before calling self.add_sample
        self.__data = OrderedDict()
        self.__labels = OrderedDict()
        self.__classes = OrderedDict()

        num_samples = len(arff_data)
        num_digits = len(str(num_samples))
        make_id = lambda index: 'row{index:0{nd}d}'.format(index=index, nd=num_digits)
        sample_classes = [cls.decode('utf-8') for cls in arff_data['class']]
        class_set = set(sample_classes)
        label_dict = dict()
        # encoding class names to labels 1 to n
        for ix, cls in enumerate(class_set):
            label_dict[cls] = ix + 1

        for index in range(num_samples):
            sample = arff_data.take([index])[0].tolist()
            sample_attrs = sample[:-1]
            sample_class = sample[-1].decode('utf-8')
            self.add_sample(sample_id=make_id(index),  # ARFF rows do not have an ID
                            features=sample_attrs,
                            label=label_dict[sample_class],
                            class_id=sample_class)
            # not necessary to set feature_names=attr_names for each sample,
            # as we do it globally after loop

        self.__feature_names = attr_names

        return