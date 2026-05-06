def wall_of_name(self):
        '''
        Appends identifiers for the different databases (such as Entrez id's)
        and returns them. Uses the CrossRef class below.
        '''
        names = []
        if self.standard_name:
            names.append(self.standard_name)
        if self.systematic_name:
            names.append(self.systematic_name)
        names.extend([xref.xrid for xref in self.crossref_set.all()])
        for i in range(len(names)):
            names[i] = re.sub(nonalpha, '', names[i])

        names_string = ' '.join(names)
        if self.standard_name:
            names_string += ' ' + re.sub(num, '', self.standard_name)
        return names_string