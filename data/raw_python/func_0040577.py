def karbasa(self, result):
        """ Finding if class probabilities are close to eachother
            Ratio of the distance between 1st and 2nd class,
            to the distance between 1st and last class.

            :param result: The dict returned by LM.calculate()
        """
        probs = result['all_probs']
        probs.sort()
        return float(probs[1] - probs[0]) / float(probs[-1] - probs[0])