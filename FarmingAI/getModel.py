import numpy as np
import joblib

def computeModel(x):
    lst = x
    lst = [float(i) for i in lst]
    lst = np.array(lst).reshape(1,-1)
    model = joblib.load(open( 'Crop_final.sav', 'rb'))
    ans = model.predict(lst)
    return ans[0]
