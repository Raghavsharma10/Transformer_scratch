def get_entropy(self, attr_name=None, attr_value=None):
        """
        Calculates the entropy of a specific attribute/value combination.
        """
        is_con = self.tree.data.is_continuous_class
        if is_con:
            if attr_name is None:
                # Calculate variance of class attribute.
                var = self._class_cdist.variance
            else:
                # Calculate variance of the given attribute.
                var = self._attr_value_cdist[attr_name][attr_value].variance
            if self.tree.metric == VARIANCE1 or attr_name is None:
                return var
            elif self.tree.metric == VARIANCE2:
                unique_value_count = len(self._attr_value_counts[attr_name])
                attr_total = float(self._attr_value_count_totals[attr_name])
                return var*(unique_value_count/attr_total)
        else:
            if attr_name is None:
                # The total number of times this attr/value pair has been seen.
                total = float(self._class_ddist.total)
                # The total number of times each class value has been seen for
                # this attr/value pair.
                counts = self._class_ddist.counts
                # The total number of unique values seen for this attribute.
                unique_value_count = len(self._class_ddist.counts)
                # The total number of times this attribute has been seen.
                attr_total = total
            else:
                total = float(self._attr_value_counts[attr_name][attr_value])
                counts = self._attr_class_value_counts[attr_name][attr_value]
                unique_value_count = len(self._attr_value_counts[attr_name])
                attr_total = float(self._attr_value_count_totals[attr_name])
            assert total, "There must be at least one non-zero count."
            
            n = max(2, len(counts))
            if self._tree.metric == ENTROPY1:
                # Traditional entropy.
                return -sum(
                    (count/total)*math.log(count/total, n)
                    for count in itervalues(counts)
                )
            elif self._tree.metric == ENTROPY2:
                # Modified entropy that down-weights universally unique values.
                # e.g. If the number of unique attribute values equals the total
                # count of the attribute, then it has the maximum amount of unique
                # values.
                return -sum(
                    (count/total)*math.log(count/total, n)
                    for count in itervalues(counts)
                #) - ((len(counts)-1)/float(total))
                ) + (unique_value_count/attr_total)
            elif self._tree.metric == ENTROPY3:
                # Modified entropy that down-weights universally unique values
                # as well as features with large numbers of values.
                return -sum(
                    (count/total)*math.log(count/total, n)
                    for count in itervalues(counts)
                #) - 100*((len(counts)-1)/float(total))
                ) + 100*(unique_value_count/attr_total)