def mixer(inputs, mix=None):
	'''
	Mix `inputs` together based on `mix` tuple

	`inputs` should be a tuple of *n* generators. 

	`mix` should be a tuple of *m* tuples, one per desired
	output channel. Each of the *m* tuples should contain
	*n* generators, corresponding to the time-sequence of 
	the desired mix levels for each of the *n* input channels.

	That is, to make an ouput channel contain a 50/50 mix of the
	two input channels, the tuple would be:

	    (constant(0.5), constant(0.5))
	
	The mix generators need not be constant, allowing for time-varying
	mix levels: 

	    # 50% from input 1, pulse input 2 over a two second cycle
	    (constant(0.5), tone(0.5))

	The mixer will return a list of *m* generators, each containing 
	the data from the inputs mixed as specified. 

	If no `mix` tuple is specified, all of the *n* input channels
	will be mixed together into one generator, with the volume of 
	each reduced *n*-fold.

	Example:

	    # three in, two out; 
	    # 10Hz binaural beat with white noise across both channels
	    mixer(
	    		(white_noise(), tone(440), tone(450)), 
	    		(
	    			(constant(.5), constant(1), constant(0)),
	    			(constant(.5), constant(0), constant(1)),
	    		)
	    	)
	'''
	if mix == None:
		# by default, mix all inputs down to one channel
		mix = ([constant(1.0 / len(inputs))] * len(inputs),)

	duped_inputs = zip(*[itertools.tee(i, len(mix)) for i in inputs])

# second zip is backwards
	return [\
			sum(*[multiply(m,i) for m,i in zip(channel_mix, channel_inputs)])\
			for channel_mix, channel_inputs in zip(mix, duped_inputs) \
			]