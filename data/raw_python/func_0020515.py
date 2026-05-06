def get_new_names_by_old():
        """Return dictionary, new label name indexed by old label name."""
        newdict = {}

        for label_type, label_names in Labels.LABEL_NAMES.items():
            for oldname in label_names[1:]:
                newdict[oldname] = Labels.LABEL_NAMES[label_type][0]
        return newdict