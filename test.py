import tensorflow as tf

def test_matmul():
    matrix1 = tf.constant([[3.,3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1,matrix2)
    print 'test_matmul():matrix1*matrix2'
    return product


def test_session():
    '''sess = tf.Session()
    result = sess.run(test_matmul())
    print 'test_session():result = sess.run()'
    sess.close()'''
    with tf.Session() as sess:
        result = sess.run(test_matmul())
        print 'test_session():result = sess.run(test_matmul())'
        return result


def test_InteractiveSession():
    sess = tf.InteractiveSession()
    x = tf.Variable([1.0, 2.0])
    y = tf.constant([3.0, 3.0])
    x.initializer.run()
    sub = tf.sub(x,y)
    print 'test_IntreactiveSession():sub = tf.sub(x,y)'
    return sub.eval()


def test_varibles():
    state = tf.Variable(0, name = 'counter')
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        print 'run(init_op):',init_op
        print sess.run(state)
        for _ in range(3):
            sess.run(update)
            print sess.run(update)


def test_fetch():
    input1 = tf.constant(2.0)
    input2 = tf.constant(3.0)
    input3 = tf.constant(5.0)
    intermed = tf.add(input2, input3)
    mul = tf.mul(input1, intermed)
    with tf.Session() as sess:
        result = sess.run([intermed, mul])
        print result


def test_feed():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.mul(input1, input2)
    with tf.Session() as sess:
        print sess.run([output], feed_dict={input1:[7.], input2:[2.]})



def show():
    print test_matmul()
    print test_session()
    print test_InteractiveSession()
    test_varibles()
    test_fetch()
    test_feed()



if __name__ == '__main__':
    show()