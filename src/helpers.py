import numpy as np 
import matplotlib.pyplot as plt 

def negate_array(x: np.ndarray):
    # Negates each element in the input array.
    # Example: input [1, 2, 3] -> output [-1, -2, -3]
    return -x  


def amazing_function(input: np.ndarray) -> np.ndarray:
    """
    Computes the function f(x) = x^2 * sin(1/x) for each element in the input array.
    
    Args:
        input (np.ndarray): A NumPy array of numerical values. 
                            The elements of the array should not be zero to avoid division by zero.
    
    Returns:
        np.ndarray: A NumPy array containing the computed values of f(x) for each element in the input array.
    """
    # Compute f(x) = x^2 * sin(1/x) for each element in the input array
    return np.sin(1 / input) * input**2



def plot_amplitude(x,y,title):
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel("time [s]")
    plt.ylabel("Amplitude")
    plt.show()
