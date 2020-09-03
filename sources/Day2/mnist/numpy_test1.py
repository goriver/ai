import numpy 

# A = numpy.array([[1,2], [3,4]])
# B = numpy.array([10, 20])
# print(numpy.dot(A, B))

A = [1,2,3,784]
B = numpy.array(A)
C = numpy.array(A, ndmin=2).T
print(A)
print(B)
print(C)

