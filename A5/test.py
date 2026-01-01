
import math
n = 32 #input size
p = 1 #padding
k = 3 #kernel
s = 2 #stride
out = math.floor((n+2*p-k)/s) + 1
print(out)