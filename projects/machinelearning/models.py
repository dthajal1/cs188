import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        # should compute the dot product of the stored weight vector and the given input, returning an nn.DotProduct object
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        dot_prod = nn.as_scalar(self.run(x))
        if (dot_prod >= 0):
            return 1
        return -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        # weights = weights + direction * multiplier

        batch_size = 1
        keep_going = True
        while keep_going:
            keep_going = False
            for x, y in dataset.iterate_once(batch_size):
                y_pred = self.get_prediction(x)
                true_label = nn.as_scalar(y)
                if y_pred != true_label:
                    self.w.update(x, true_label)
                    keep_going = True                

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        # simple two-layer neural network for mapping an input row vector x to an output vector f(x)
        # f(x) = relu(x * W1 + b1) * W2 + b2
        # (x * W1 + b1) = (i, 1) * (1, h) + (h, 1) = (i, h) + (h, 1)
        # parameter matrices: W1 (i by h) and W2 
        # parameter vectors: b1 (h by 1) and b2
        # where i = dim of input vector x = batch_size
        # h = hidden layer size
        hidden_layer_size = 100
        self.W1 = nn.Parameter(1, hidden_layer_size)
        self.W2 = nn.Parameter(hidden_layer_size, 1)
        self.b1 = nn.Parameter(1, hidden_layer_size)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # f(x) = relu(x * W1 + b1) * W2 + b2
        # (1 by 1) * (1 by 100) = (1 by 100)
        # (1 by 100) + (1 by 100) = (1 by 100)

        # (1 by 100) * (100 by 1) = (1 by 1)
        # (1 by 1) + (1 by 1) = (1 by 1)

        x_w1 = nn.Linear(x, self.W1)
        input = nn.AddBias(x_w1, self.b1)
        relu = nn.ReLU(input)
        relu_w2 = nn.Linear(relu, self.W2)
        output = nn.AddBias(relu_w2, self.b2)
        return output       

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        y_pred = self.run(x)
        loss = nn.SquareLoss(y_pred, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 1
        learning_rate = -0.001
        loss = float('inf')
        while loss > 0.02:
            for x, y in dataset.iterate_once(batch_size):
                params = [self.W1, self.W2, self.b1, self.b2]
                grad_wrt_W1, grad_wrt_W2, grad_wrt_b1, grad_wrt_b2 = nn.gradients(self.get_loss(x, y), params)
                self.W1.update(grad_wrt_W1, learning_rate)
                self.W2.update(grad_wrt_W2, learning_rate)
                self.b1.update(grad_wrt_b1, learning_rate)
                self.b2.update(grad_wrt_b2, learning_rate)

                loss = nn.as_scalar(self.get_loss(x, y))



class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        # f(x) = relu(x * W1 + b1) * W2 + b2
        # (1 by 784) * (784 by 100) = (1 by 100)
        # (1 by 100) + (1 by 100) = (1 by 100)

        # (1 by 100) * (100 by 10) = (1 by 10)
        # (1 by 10) + (1 by 10) = (1 by 10)
        hidden_layer_size = 300
        self.W1 = nn.Parameter(784, hidden_layer_size)
        self.W2 = nn.Parameter(hidden_layer_size, 10)
        self.b1 = nn.Parameter(1, hidden_layer_size)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # f(x) = relu(x * W1 + b1) * W2 + b2
        # (1 by 784) * (784 by 100) = (1 by 100)
        # (1 by 100) + (1 by 100) = (1 by 100)

        # (1 by 100) * (100 by 10) = (1 by 10)
        # (1 by 10) + (1 by 10) = (1 by 10)

        x_w1 = nn.Linear(x, self.W1)
        input = nn.AddBias(x_w1, self.b1)
        relu = nn.ReLU(input)
        relu_w2 = nn.Linear(relu, self.W2)
        output = nn.AddBias(relu_w2, self.b2)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        logits = self.run(x)
        true_labels = y
        return nn.SoftmaxLoss(logits, true_labels)

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 200
        learning_rate = -0.09
        validation_accuracy = 0
        while validation_accuracy <= 0.98:
            for x, y in dataset.iterate_once(batch_size):
                params = [self.W1, self.W2, self.b1, self.b2]
                grad_wrt_W1, grad_wrt_W2, grad_wrt_b1, grad_wrt_b2 = nn.gradients(self.get_loss(x, y), params)
                self.W1.update(grad_wrt_W1, learning_rate)
                self.W2.update(grad_wrt_W2, learning_rate)
                self.b1.update(grad_wrt_b1, learning_rate)
                self.b2.update(grad_wrt_b2, learning_rate)

                validation_accuracy = dataset.get_validation_accuracy()

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        # h_i = relu(z_(i - 1)  + b)
        # z_0 = x_0 * W
        # z_i = x_i * W + h_i * W_hidden

        # x_0 = (1 by self.num_chars)
        # W = (self.num_chars by 100)
        # (1 by 100)

        # h_i = (1 by d)
        # W_hidden = (d by 100)
        # (1 by 100)

        hidden_layer_size = 300

        self.W = nn.Parameter(self.num_chars, hidden_layer_size)
        self.W_hidden = nn.Parameter(hidden_layer_size, hidden_layer_size)

        self.b1 = nn.Parameter(1, hidden_layer_size)
        self.b2 = nn.Parameter(hidden_layer_size, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        # h_i = relu(z_(i - 1)  + b)
        # z_0 = x_0 * W
        # z_i = x_i * W + h_i * W_hidden

        z_prev = None
        for i, x in enumerate(xs):
            if i == 0:
                z = nn.Linear(x, self.W)
                z_prev = z
            else:
                xi_W = nn.Linear(x, self.W)

                h_i = nn.ReLU(nn.AddBias(z_prev, self.b1))

                hi_Whidden = nn.Linear(h_i, self.W_hidden)

                z = nn.Add(xi_W, hi_Whidden)
                z_prev = z
        return nn.Linear(z, self.b2)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        logits = self.run(xs)
        true_labels = y
        return nn.SoftmaxLoss(logits, true_labels)

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 100
        learning_rate = -0.08
        validation_accuracy = 0
        while validation_accuracy <= 0.85:
            for x, y in dataset.iterate_once(batch_size):
                params = [self.W, self.W_hidden, self.b1, self.b2]
                grad_wrt_W, grad_wrt_W_hidden, grad_wrt_b1, grad_wrt_b2 = nn.gradients(self.get_loss(x, y), params)
                self.W.update(grad_wrt_W, learning_rate)
                self.W_hidden.update(grad_wrt_W_hidden, learning_rate)
                self.b1.update(grad_wrt_b1, learning_rate)
                self.b2.update(grad_wrt_b2, learning_rate)

                validation_accuracy = dataset.get_validation_accuracy()
        