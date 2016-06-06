import tensorflow.examples.tutorials.mnist.input_data as input_data

def download():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

if __name__ == '__main__':
    download()
