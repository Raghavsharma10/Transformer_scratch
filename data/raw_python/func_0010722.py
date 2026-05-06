def render(self, **context):
		"""
		Render this template by applying it to `context`.
		@params:
			`context`: a dictionary of values to use in this rendering.
		@returns:
			The rendered string
		"""
		# Make the complete context we'll use.
		localns = self.envs.copy()
		localns.update(context)

		try:
			exec(str(self.code), None, localns)
			return localns[Liquid.COMPLIED_RENDERED_STR]
		except Exception:
			stacks = list(reversed(traceback.format_exc().splitlines()))
			for stack in stacks:
				stack = stack.strip()
				if stack.startswith('File "<string>"'):
					lineno = int(stack.split(', ')[1].split()[-1])
					source = []
					if 'NameError:' in stacks[0]:
						source.append('Do you forget to provide the data?')

					import math
					source.append('\nCompiled source (use debug mode to see full source):')
					source.append('---------------------------------------------------')
					nlines = len(self.code.codes)
					nbit   = int(math.log(nlines, 10)) + 3
					for i, line in enumerate(self.code.codes):
						if i - 7 > lineno or i + 9 < lineno: continue
						if i + 1 != lineno:
							source.append('  ' + (str(i+1) + '.').ljust(nbit) + str(line).rstrip())
						else:
							source.append('* ' + (str(i+1) + '.').ljust(nbit) + str(line).rstrip())

					raise LiquidRenderError(
						stacks[0], 
						repr(self.code.codes[lineno - 1]) + 
						'\n' + '\n'.join(source) + 
						'\n\nPREVIOUS EXCEPTION:\n------------------\n' + 
						'\n'.join(stacks) + '\n' +
						'\nCONTEXT:\n------------------\n' +
						'\n'.join(
							'  ' + key + ': ' + str(val) 
							for key, val in localns.items() if not key.startswith('_liquid_') and not key.startswith('__')
						) + '\n'
					)
			raise