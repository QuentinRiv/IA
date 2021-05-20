
# Construct an artificial test set which cover the whole plane


def grid(func, xRange, yRange, step, …):
    """ Function to construct an artificial test set which cover the whole planen,  given  a range (xRange, yRange) and with the given precision (step). """

    # Create meshgrid
    x = np.arange(xRange[0], xRange[1], step)
    y = np.arange(yRange[0], yRange[1], step)
    X, Y = np.meshgrid(x, y)

    return X, Y

# Hint: check numpy.meshgrid function https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html#numpy.meshgrid


# Apply your prediction function to the grid points (don’t forget when we predict the data set doesn’t contain the class labels )

Y_predicted = prediction_func(X, Y, )

# Create a function that draws a plane on the instances
# using the  first two attributes only, and color them according
# to the class to which they belong.


def draw_plane(X, Y, Y_predicted, levels …)


Hint: see the plot.countour() function https: // matplotlib.org / api / _as_gen / matplotlib.pyplot.contour.html
