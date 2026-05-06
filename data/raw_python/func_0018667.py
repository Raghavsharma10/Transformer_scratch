def starts_with(self, other: 'Key') -> bool:
        """
        Checks if this key starts with the other key provided. Returns False if key_type, identity
        or group are different.
        For `KeyType.TIMESTAMP` returns True.
        For `KeyType.DIMENSION` does prefix match between the two dimensions property.
        """
        if (self.key_type, self.identity, self.group) != (other.key_type, other.identity,
                                                          other.group):
            return False
        if self.key_type == KeyType.TIMESTAMP:
            return True
        if self.key_type == KeyType.DIMENSION:
            if len(self.dimensions) < len(other.dimensions):
                return False
            return self.dimensions[0:len(other.dimensions)] == other.dimensions