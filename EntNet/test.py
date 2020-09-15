f= open("12.txt","a+")
for i in range(100):
     f.write("Append line %d\r\n" % (i+1))
