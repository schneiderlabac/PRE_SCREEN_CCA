import numpy as np 
from src.helpers import negate_array, amazing_function,plot_amplitude
# or simply import all with: 
#from src.helpers import *



if __name__ == "__main__":
    # Load or generate data 
    x = np.linspace(1e-10, 0.05,num=1000)
    # Apply functions
    y = amazing_function(x)
    y_neg = negate_array(y)
    
    # Create plot
    plot_amplitude(x=x, y=y_neg, title="amazing function")
    exit()