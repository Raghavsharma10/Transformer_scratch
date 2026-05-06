def _process_flat_kwargs(source, kwargs):
		"""Apply a flat namespace transformation to recreate (in some respects) a rich structure.
		
		This applies several transformations, which may be nested:
		
		`foo` (singular): define a simple value named `foo`
		`foo` (repeated): define a simple value for placement in an array named `foo`
		`foo[]`: define a simple value for placement in an array, even if there is only one
		`foo.<id>`: define a simple value to place in the `foo` array at the identified index
		
		By nesting, you may define deeper, more complex structures:
		
		`foo.bar`: define a value for the named element `bar` of the `foo` dictionary
		`foo.<id>.bar`: define a `bar` dictionary element on the array element marked by that ID
		
		References to `<id>` represent numeric "attributes", which makes the parent reference be treated as an array,
		not a dictionary. Exact indexes might not be able to be preserved if there are voids; Python lists are not
		sparse.
		
		No validation of values is performed.
		"""
		
		ordered_arrays = []
		
		# Process arguments one at a time and apply them to the kwargs passed in.
		
		for name, value in source.items():
			container = kwargs
			
			if '.' in name:
				parts = name.split('.')
				name = name.rpartition('.')[2]
				
				for target, following in zip(parts[:-1], parts[1:]):
					if following.isnumeric():  # Prepare any use of numeric IDs.
						container.setdefault(target, [{}])
						if container[target] not in ordered_arrays:
							ordered_arrays.append(container[target])
						container = container[target][0]
						continue
					
					container = container.setdefault(target, {})
			
			if name.endswith('[]'):  # `foo[]` or `foo.bar[]` etc.
				name = name[:-2]
				container.setdefault(name, [])
				container[name].append(value)
				continue
			
			if name.isnumeric() and container is not kwargs:  # trailing identifiers, `foo.<id>`
				container[int(name)] = value
				continue
			
			if name in container:
				if not isinstance(container[name], list):
					container[name] = [container[name]]
				
				container[name].append(value)
				continue
			
			container[name] = value
		
		for container in ordered_arrays:
			elements = container[0]
			del container[:]
			container.extend(value for name, value in sorted(elements.items()))