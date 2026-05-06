def _findNextPrime(self, N):
        """Generate the first N primes"""
        primes = self.primes
        nextPrime = primes[-1]+1
        while(len(primes)<N):
            maximum = nextPrime * nextPrime
            prime = 1
            for i in primes:
                if i > maximum:
                    break
                if nextPrime % i == 0:
                    prime = 0
                    break
            if prime:
                primes.append(nextPrime)
            nextPrime+=1