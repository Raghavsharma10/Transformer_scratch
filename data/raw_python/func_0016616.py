def split(self, ratio=0.5, leave_one_out=False):
        """
        Returns two Data instances, containing the data randomly split between
        the two according to the given ratio.
        
        The first instance will contain the ratio of data specified.
        The second instance will contain the remaining ratio of data.
        
        If leave_one_out is True, the ratio will be ignored and the first
        instance will contain exactly one record for each class label, and
        the second instance will contain all remaining data.
        """
        a_labels = set()
        a = self.copy_no_data()
        b = self.copy_no_data()
        for row in self:
            if leave_one_out and not self.is_continuous_class:
                label = row[self.class_attribute_name]
                if label not in a_labels:
                    a_labels.add(label)
                    a.data.append(row)
                else:
                    b.data.append(row)
            elif not a:
                a.data.append(row)
            elif not b:
                b.data.append(row)
            elif random.random() <= ratio:
                a.data.append(row)
            else:
                b.data.append(row)
        return a, b