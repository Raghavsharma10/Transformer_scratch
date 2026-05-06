def get_weighted_random_value(self):
		"""
		This will generate a value by creating a cumulative distribution, 
		and a random number, and selecting the value who's cumulative 
		distribution interval contains the generated random number. 
		
		For example, if there's 0.7 chance of generating the letter "a"
		and 0.3 chance of generating the letter "b", then if you were to 
		pick one letter 100 times over, the number of a's and b's you 
		would have are likely to be around 70 and 30 respectively.
		
		The mechanics are known as "Cumulative distribution functions"
		(https://en.wikipedia.org/wiki/Cumulative_distribution_function)
		"""
		from bisect import bisect
		from random import random
		#http://stackoverflow.com/questions/4437250/choose-list-variable-given-probability-of-each-variable
		
		total = sum(self.values())
		
		P = [(k, (v / float(total))) for k, v in self.items()]
		
		cdf = [P[0][1]]
		for i in range(1, len(P)):
			cdf.append(cdf[-1] + P[i][1])
			
		return P[bisect(cdf, random())][0]