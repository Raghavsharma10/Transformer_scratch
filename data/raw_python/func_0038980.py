def label_set(self):
        """Set of labels in the dataset corresponding to class_set."""
        label_set = list()
        for class_ in self.class_set:
            samples_in_class = self.sample_ids_in_class(class_)
            label_set.append(self.labels[samples_in_class[0]])

        return label_set