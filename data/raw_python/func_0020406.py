def calculate_response(self, challenge, password):
        """Calculate response for the challenge-response authentication"""
        to_hash = (challenge + "-" + password).encode("UTF-16LE")
        hashed = hashlib.md5(to_hash).hexdigest()
        return "{0}-{1}".format(challenge, hashed)