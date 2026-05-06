def _enableTracesVerilog(self, verilogFile):
        ''' Enables traces in a Verilog file'''
        fname, _ = os.path.splitext(verilogFile)
        inserted = False
        for _, line in enumerate(fileinput.input(verilogFile, inplace = 1)):
            sys.stdout.write(line)
            if line.startswith("end") and not inserted:
                sys.stdout.write('\n\n') 
                sys.stdout.write('initial begin\n')
                sys.stdout.write('    $dumpfile("{}_cosim.vcd");\n'.format(fname)) 
                sys.stdout.write('    $dumpvars(0, dut);\n') 
                sys.stdout.write('end\n\n') 
                inserted = True