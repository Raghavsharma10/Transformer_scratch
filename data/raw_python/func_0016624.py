def predict(self, record, depth=0):
        """
        Returns the estimated value of the class attribute for the given
        record.
        """
        
        # Check if we're ready to predict.
        if not self.ready_to_predict:
            raise NodeNotReadyToPredict
        
        # Lookup attribute value.
        attr_value = self._get_attribute_value_for_node(record)
        
        # Propagate decision to leaf node.
        if self.attr_name:
            if attr_value in self._branches:
                try:
                    return self._branches[attr_value].predict(record, depth=depth+1)
                except NodeNotReadyToPredict:
                    #TODO:allow re-raise if user doesn't want an intermediate prediction?
                    pass
                
        # Otherwise make decision at current node.
        if self.attr_name:
            if self._tree.data.is_continuous_class:
                return self._attr_value_cdist[self.attr_name][attr_value].copy()
            else:
#                return self._class_ddist.copy()
                return self.get_value_ddist(self.attr_name, attr_value)
        elif self._tree.data.is_continuous_class:
            # Make decision at current node, which may be a true leaf node
            # or an incomplete branch in a tree currently being built.
            assert self._class_cdist is not None
            return self._class_cdist.copy()
        else:
            return self._class_ddist.copy()