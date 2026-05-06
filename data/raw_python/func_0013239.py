def get_ok_results(self, verbose=True):
        """Return a list of results with ok status
        """
        if len(self.trials) == 0:
            return []

        not_ok = np.where(np.array(self.statuses()) != "ok")[0]

        if len(not_ok) > 0 and verbose:
            print("{0}/{1} trials were not ok.".format(len(not_ok), len(self.trials)))
            print("Trials: " + str(not_ok))
            print("Statuses: " + str(np.array(self.statuses())[not_ok]))

        r = [merge_dicts({"tid": t["tid"]}, t["result"].to_dict())
             for t in self.trials if t["result"]["status"] == "ok"]
        return r