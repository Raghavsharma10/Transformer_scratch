def hash(self, value):
        """
            function hash() implement to acquire hash value that use simply method that weighted sum.

            Parameters:
            -----------
            value: string
                the value is param of need acquire hash
            Returns:
            --------
            result
                hash code for value
        """
        result = 0
        for i in range(len(value)):
            result += self.seed * result + ord(value[i])
        return (self.capacity - 1) % result