import torch

class ConvModelClass(torch.nn.Module):

    def __init__(self):

        super(ConvModelClass, self).__init__()

        self.output_size = 4

        self.scanning_layers = torch.nn.Sequential(
           
            # This is a convolutional layer that takes in 7x7 pixels at a time and slides 4 pixels at a time
            # The out_channels multiplies the number of nodes in the layer
            torch.nn.Conv2d(in_channels=4, out_channels=512, kernel_size=7, stride=4),
            # Choose an activation function.
            torch.nn.ReLU(),

            # Add some regularization between layers as well, using either BatchNorm or Dropout
            torch.nn.BatchNorm2d(512),

            # Add at least 2 more sequences of convolution layers, activations, and regularization.
            # Channels must link with previous layer similar to how linear layers link in shape
            # But don't confuse channels with the size of the image, as the image size is handled automatically
            torch.nn.Conv2d(512, 1024, 7, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),


            torch.nn.Conv2d(1024, 256, 5, 3),
            torch.nn.ReLU(),


            # Pooling Layer combines cells in some functional way. The following layer will have a smaller image size.
            torch.nn.MaxPool2d(5),
          
        )

        # These will be linear layers leading up to your output
        self.classification_layers = torch.nn.Sequential(
            # Our Convolutional layers 
            torch.nn.Flatten(),

            torch.nn.Linear(256, 1024),
            torch.nn.ReLU(),

            # Final (output) layer
            # You might have to guess & check the input size
            torch.nn.Linear(1024, self.output_size)
        )      

    def forward(self, x):

        z = self.scanning_layers(x)

        out = self.classification_layers(z)

        return out