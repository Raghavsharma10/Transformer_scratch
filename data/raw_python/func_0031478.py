def get_ec_numbers(cls, entry):
        """
        get list of models.ECNumber objects from XML node entry

        :param entry:  XML node entry
        :return: list of models.ECNumber objects
        """
        ec_numbers = []

        for ec in entry.iterfind("./protein/recommendedName/ecNumber"):
            ec_numbers.append(models.ECNumber(ec_number=ec.text))
        return ec_numbers