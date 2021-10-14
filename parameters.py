# Set important parameters
learning_rate = 1e-2
inner_loop_iterations = 32
outer_loop_iterations = 5
num_classes = 10

noise_size = 64     # size of noise or curriculum vector
img_size = 28    # width / height of generated image

inner_loop_batch_size = 128
outer_loop_batch_size = 128

mnist_mean = 0.1307         # for normalizing mnist images
mnist_std = 0.3081          # for normalizing mnist images
use_curriculum = True
num_architectures = 10


imgs_per_row = num_classes
