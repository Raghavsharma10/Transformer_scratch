def dump(self):
        """Print an overview of the ARFF file."""
        print("Relation " + self.relation)
        print("  With attributes")
        for n in self.attributes:
            if self.attribute_types[n] != TYPE_NOMINAL:
                print("    %s of type %s" % (n, self.attribute_types[n]))
            else:
                print("    " + n + " of type nominal with values " + ', '.join(self.attribute_data[n]))
        for d in self.data:
            print(d)