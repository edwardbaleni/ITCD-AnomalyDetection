# Essentially, I want to look for an npy file
# If that is not available then I want to 
# call upon the data file to create npy file, and open data here


# https://stackoverflow.com/questions/48376580/how-to-read-data-in-google-colab-from-my-google-drive
# 
from google.colab import drive
drive.mount('/content/drive')