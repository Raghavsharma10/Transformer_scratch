def get_block_bounds(filename):
 """Pre block starts
    start 0-indexted, end 1-indexted

 :param filename: filename
 :type filename: string
 :return: 0-index start and 1-index end
 :rtype: array of arrays with the [start end] of each block
 """
 bs = []
 with open(filename,'rb') as inf:
  while True:
   bytes1 = inf.read(12)
   if len(bytes1) < 12: break
   bs.append([inf.tell()-12])
   gzip_id1,gzip_id2,compression_method,flag,mtime,xfl,osval,xlen=struct.unpack('<BBBBIBBH',bytes1)
   # ready to look in extra field
   bytes2 = inf.read(xlen) # all the extra field stuff 
   s = StringIO.StringIO(bytes2)
   obsslen = 0
   blocksize = 0
   while True:
     v1 = s.read(4)
     if len(v1) == 0: 
       break
     if len(v1) < 4:
       sys.stderr.write("lack header values ERROR\n")
       return False
     s1,s2,slen = struct.unpack('<BBH',v1)
     if s1 == 66 and s2 == 67:
       has_id = True
       obsslen = slen
       blocksize = struct.unpack('<H',s.read(slen))[0]
     else:
       v = s.read(slen)
   chunk = inf.read(blocksize-1-xlen-19)
   inf.read(9)
   bs[-1].append(inf.tell())
 return bs