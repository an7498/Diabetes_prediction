# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
loaded_model = pickle.load(open('F:/OneDrive/Desktop/ml project/finalized_model.sav', 'rb'))
#making a predictive system
input_data=(10,168,74,0,0,38,0.537,34)
#change input data as numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
#standarize the input data


prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]==0):
  print(' The person is not diabetic')
else:
  print('The person is diabetic')