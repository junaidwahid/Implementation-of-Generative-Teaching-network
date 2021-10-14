import torchvision
def generate_img(img_tensor, mnist_std, mnist_mean):
    '''
    Function that renders an MNIST image.
    '''
    return torchvision.transforms.ToPILImage()(1 - ((img_tensor * mnist_std) + mnist_mean))