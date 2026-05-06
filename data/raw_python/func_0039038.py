def generate_random_string(cls, length):
        """
        Generatesa a [length] characters alpha numeric secret
        """
        # avoid things that could be mistaken ex: 'I' and '1'
        letters = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
        return "".join([random.choice(letters) for _ in range(length)])