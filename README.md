
# Generative Teaching network
GTNs are deep neural networks that generate data and/or training environments that a learner (e.g. a freshly initialized neural network) trains on for a few SGD steps before being tested on a target task. Then it differentiates through the entire learning process via meta-gradients to update the GTN parameters to improve performance on the target task. GTNs have the beneficial property that they can theoretically generate any type of data or training environment, making their potential impact large. One of the exciting applications of GTNs is accelerating the evaluation of candidate architectures for Neural Architecture Search (NAS).

This implementation uses the MNIST dataset for demonstration purposes and showcasing the power of GTNs in finding optimal architecture for the task in a faster way.

## Architecture of GTN
There are two basic component of GTN: Generator and Classifier
### Generator
The generator learns to generate synthetic data for a particular task. The generator consist of two fully connected layer following two convolutional layers
  
### Classifier
The classifier utilizes the data generated by the generator. The classifier is designed to generate a random number of convolutional filters in both layers Convolutional layers so the teacher(Generator) generalizes to other architectures in NAS.

## Training

After training the classifier on the generated data, the classifier will be evaluated on a batch of real data and backpropagate the resulting loss to the generator. This process will continue on a number of iterations.

Only after 50 outer iterations(generator update iterations), the generators start producing examples that help the classifier to reach 90% validation with only 32 iterations which is amazing. This is pretty cool and this is what we need in order to perform a wide search in neural architecture search. The power of training a new architecture so fast really boosts the overall architecture search.

Another interesting point, the data generated by generator does not look realistic at all. But, it has learned the art of generating data which would help the classifier to reach maximum accuracy in fewer iterations.

![Generator data](https://user-images.githubusercontent.com/16369846/135759541-74635cc1-2533-4a58-84c1-c78fb82f1c51.png)



## Neural Architecture Search
Once trained on many iterations, our generator becomes so good to produce such specific data that helps any randomly initialized model to converge in a few iterations. This helps us to perform NAS efficiently and faster. The classifier class has properties to generate a random number of convolutional filters in every new initialization. Due to which the NAS has been performed on different random architecture in every iteration. Due to the power of the Generator in generating specific data for the given task, the train time of classifier becomes less that shortens the overall architecture search time. 

![image](https://user-images.githubusercontent.com/16369846/135759722-f32dfc13-ba7c-4c88-bd3e-3f202b89f504.png)