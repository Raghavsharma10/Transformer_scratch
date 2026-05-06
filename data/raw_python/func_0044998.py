def rate_limit(max_interval=20, sampling=3, f=lambda x: x):
	'''x rises by 1 from 0 on each iteraton, back to 0 on triggering.
		f(x) should rise up to f(max_interval) in some way (with default
			"f(x)=x" probability rises lineary with 100% chance on "x=max_interval").
		"sampling" affect probablility in an "c=1-(1-c0)*(1-c1)*...*(1-cx)" exponential way.'''
	from random import random
	val = 0
	val_max = float(f(max_interval))
	while True:
		if val % sampling == 0:
			trigger = random() > (val_max - f(val)) / val_max
			if trigger: val = 0
			yield trigger
		else: yield False
		val += 1