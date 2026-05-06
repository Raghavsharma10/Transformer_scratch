def get_value(self) -> ScalarType:
        """Returns the value of a Scalar node.

        Use is_scalar(type) to check which type the node has.
        """
        if self.yaml_node.tag == 'tag:yaml.org,2002:str':
            return self.yaml_node.value
        if self.yaml_node.tag == 'tag:yaml.org,2002:int':
            return int(self.yaml_node.value)
        if self.yaml_node.tag == 'tag:yaml.org,2002:float':
            return float(self.yaml_node.value)
        if self.yaml_node.tag == 'tag:yaml.org,2002:bool':
            return self.yaml_node.value in ['TRUE', 'True', 'true']
        if self.yaml_node.tag == 'tag:yaml.org,2002:null':
            return None
        raise RuntimeError('This node with tag {} is not of the right type'
                           ' for get_value()'.format(self.yaml_node.tag))