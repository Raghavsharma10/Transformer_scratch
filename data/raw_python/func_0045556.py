def first_order_gradient(arr, axis=0):
  '''
  Returns the gradient of arr along a particular axis using
  the first order forward-difference.
  Additionally, the result is padded with 0 so that the
  returned array is the same shape as in input array.
  '''
  grad_arr = difference(arr, n=1, axis=axis)
  return np.insert(grad_arr, grad_arr.shape[axis], 0, axis=axis)