def check_answer(self, hash_answer):
        """ Check if the returned hash is in our challenges list.

        :param hash_answer: Hash that we compare to our list of challenges
        :return: boolean indicating if answer is correct, True, or not, False
        """
        for challenge in self.challenges:
            if challenge.response == hash_answer:
                # If we don't discard a used challenge then a node
                # could fake having the file because it already
                # knows the proper response
                self.delete_challenge(hash_answer)
                return True
        return False