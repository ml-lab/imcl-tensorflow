import tensorflow as tf

def test_matmul():
    matrix1 = tf.constant([[3.,3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1,matrix2)
    print 'test_matmul():matrix1*matrix2'
    return product

def test_session():
    sess = tf.Session()
    result = sess.run(test_matmul())
    print 'test_session():result = sess.run()'
    sess.close()
    return result

def show():
    print test_matmul()
    print test_session()

if __name__ == '__main__':
    show()