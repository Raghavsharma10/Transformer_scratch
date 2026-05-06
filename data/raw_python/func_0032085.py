def make_spark_lines(table,filename,sc,**kwargs):
	spark_output = True
	lines_out_count = False
	extrema = False
	for key,value in kwargs.iteritems():
		if key == 'lines_out_count':
			lines_out_count = value
		if key == 'extrema':
			extrema = value
	# removing datetime references from imported postgis database
	# CURRENTLY datetime from postgis dbs throw errors 
	# fields containing dates removed
	list = []
	count = 0
	for row in table.columns.values.tolist():
		if 'date' in row:
			list.append(count)
		count += 1

	table.drop(table.columns[list], axis=1, inplace=True)


	# getting spark arguments
	if lines_out_count == False:
		args = make_spark_args(table,25,lines_out = True,extrema=extrema)
	else:
		args = make_spark_args(table,25,lines_out_count=lines_out_count)
	# concurrent represents rdd structure that will be parrelized
	concurrent = sc.parallelize(args)

	# getting table that would normally be going into this function
	table = concurrent.map(map_spark_lines).collect()



	'''
	alignment_field = False
	spark_output = True
	if kwargs is not None:
		for key,value in kwargs.iteritems():
			if key == 'alignment_field':
				alignment_field = value 
			if key == 'spark_output':
				spark_output = value

	#changing dataframe to list if dataframe
	if isinstance(table,pd.DataFrame):
		table=df2list(table)
	header=table[0]
	total = []
	# making table the proper iterable for each input 
	if spark_output == True:
		#table = sum(table,[])
		pass
	else:
		table = table[1:]
	'''
	'''
	# making filenames list
	filenames = []
	count = 0
	while not len(filenames) == len(table):
		count += 1
		filename = 'lines%s.geojson' % str(count)
		filenames.append(filename)

	args = []
	# zipping arguments together for each value in table
	for filename,row in itertools.izip(filenames,table):
		args.append([filename,row])


	concurrent = sc.parallelize(args)
	concurrent.map(map_lines_output).collect()
	'''
	'''
	count=0
	total=0
	for row in table:
		count+=1
		# logic to treat rows as outputs of make_line or to perform make_line operation
		if spark_output == False:
			value = make_line([header,row],list=True,postgis=True,alignment_field=alignment_field)
		elif spark_output == True:
			value = row

		# logic for how to handle starting and ending geojson objects
		if row==table[0]:
			#value=make_line([header,row],list=True,postgis=True,alignment_field=alignment_field)
			if not len(table)==2:
				value=value[:-3]
				totalvalue=value+['\t},']
		
		elif row==table[-1]:
			#value=make_line([header,row],list=True,postgis=True,alignment_field=alignment_field)
			value=value[2:]
			totalvalue=totalvalue+value
		else:
			#value=make_line([header,row],list=True,postgis=True,alignment_field=alignment_field)
			value=value[2:-3]
			value=value+['\t},']
			totalvalue=totalvalue+value
		if count == 1000:
			total += count
			count = 0
			print '[%s/%s]' % (total,len(table))
	bl.parselist(totalvalue,filename)
	'''