# %%
import numpy as np
# Load in Performance matrix, P
    # This is obtained from hyper


# TODO: Do this in separate IPM folder!
# Load in internal Performance, IPM measures.
    # Use the results from tuning to get O 
    # Use O to get the IPM measures
    # Save the IPM measures

# extract meta-features per task g(X)
from sklearn.datasets import load_iris
from pymfe.mfe import MFE

data = load_iris()
X = data.data

mfe = MFE()
mfe.fit(X)
ft = mfe.extract()
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1]))) 
# TODO: make sure to make all NAs 0
meta_vals = np.nan_to_num(ft[1], copy=False)
meta_feat = ft[0]


# Train the proxy performance evaluator f(.)

# Train the meta-surrogate funcion h(I, g)

# Save
    # meta-feature extractor
    # IPM extractor
    # PPE
    # MSF




# %%