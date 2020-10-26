import tensorflow as tf

#Notice: tf.constant([2], name="constant_a") creates a new tf.Operation named "constant_a",
#and returns a tf.Tensor named "constant_a:0".
a = tf.constant([2], name = 'constant_a') 
b = tf.constant([3], name = 'constant_b')

print(a)

tf.print(a.numpy()[0])

@tf.function
def add(a,b):
    c = tf.add(a, b)
    #c = a + b is also a way to define the sum of the terms
    print(c)
    return c

result = add(a,b)
tf.print(result[0])

#Defining multidimensional arrays
Scalar = tf.constant(2)
Vector = tf.constant([5,6,2])
Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )

print ("Scalar (1 entry):\n %s \n" % Scalar)

print ("Vector (3 entries) :\n %s \n" % Vector)

print ("Matrix (3x3 entries):\n %s \n" % Matrix)

print ("Tensor (3x3x3 entries) :\n %s \n" % Tensor)

Scalar.shape
Tensor.shape

#play with old fctions
Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

@tf.function
def add():
    add_1_operation = tf.add(Matrix_one, Matrix_two)
    return add_1_operation



print ("Defined using tensorflow function :")
add_1_operation = add()
print(add_1_operation)
print ("Defined using normal expressions :")
add_2_operation = Matrix_one + Matrix_two
print(add_2_operation)


#multiply matrix
Matrix_one = tf.constant([[2,3],[3,4]])
Matrix_two = tf.constant([[2,3],[3,4]])

@tf.function
def mathmul():
  return tf.matmul(Matrix_one, Matrix_two)


mul_operation = mathmul()

print ("Defined using tensorflow function :")
print(mul_operation)

#when you define a variable, TensorFlow adds a tf.Operation to your graph. 
#Then, this operation will store a writable tensor value. So, you can update the value of a variable through each run.
#Let's first create a simple counter, by first initializing a variable v that will be increased one unit at a time:

v = tf.Variable(0)

@tf.function
def increment_by_one(v):
        v = tf.add(v,1)
        return v

for i in range(3):
    v = increment_by_one(v)
    print(v)
    
#operations
a = tf.constant([5])
b = tf.constant([2])
c = tf.add(a,b)
d = tf.subtract(a,b)


print ('c =', c)
    
print ('d =', d)




