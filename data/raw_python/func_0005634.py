def get_registered_names(mcs):
        """ Return's list of names (keys) registered in this tree.
            For each name specific classes exists
        """
        return [k for k, v in six.iteritems(mcs._base_classes_hash) if v]