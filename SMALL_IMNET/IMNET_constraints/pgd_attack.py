import numpy as np
import tensorflow as tf


from compat import reduce_max, reduce_sum, softmax_cross_entropy_with_logits


def op_with_scalar_cast(a, b, f):
  """
  Builds the graph to compute f(a, b).
  If only one of the two arguments is a scalar and the operation would
  cause a type error without casting, casts the scalar to match the
  tensor.
  :param a: a tf-compatible array or scalar
  :param b: a tf-compatible array or scalar
  """

  try:
    return f(a, b)
  except (TypeError, ValueError):
    pass

  def is_scalar(x):
    """Return True if `x` is a scalar"""
    if hasattr(x, "get_shape"):
      shape = x.get_shape()
      return shape.ndims == 0
    if hasattr(x, "ndim"):
      return x.ndim == 0
    assert isinstance(x, (int, float))
    return True

  a_scalar = is_scalar(a)
  b_scalar = is_scalar(b)

  if a_scalar and b_scalar:
    raise TypeError("Trying to apply " + str(f) + " with mixed types")

  if a_scalar and not b_scalar:
    a = tf.cast(a, b.dtype)

  if b_scalar and not a_scalar:
    b = tf.cast(b, a.dtype)

  return f(a, b)


def zero_out_clipped_grads(grad, x, clip_min, clip_max):
  """
  Helper function to erase entries in the gradient where the update would be
  clipped.
  :param grad: The gradient
  :param x: The current input
  :param clip_min: Minimum input component value
  :param clip_max: Maximum input component value
  """
  signed_grad = tf.sign(grad)

  # Find input components that lie at the boundary of the input range, and
  # where the gradient points in the wrong direction.
  clip_low = tf.logical_and(tf.less_equal(x, tf.cast(clip_min, x.dtype)),
                            tf.less(signed_grad, 0))
  clip_high = tf.logical_and(tf.greater_equal(x, tf.cast(clip_max, x.dtype)),
                             tf.greater(signed_grad, 0))
  clip = tf.logical_or(clip_low, clip_high)
  grad = tf.where(clip, mul(grad, 0), grad)

  return grad



def mul(a, b):
  """
  A wrapper around tf multiplication that does more automatic casting of
  the input.
  """
  def multiply(a, b):
    """Multiplication"""
    return a * b
  return op_with_scalar_cast(a, b, multiply)      


def div(a, b):
  """
  A wrapper around tf division that does more automatic casting of
  the input.
  """
  def divide(a, b):
    """Division"""
    return a / b
  return op_with_scalar_cast(a, b, divide)



def clip_by_value(t, clip_value_min, clip_value_max, name=None):
  """
  A wrapper for clip_by_value that casts the clipping range if needed.
  """
  def cast_clip(clip):
    """
    Cast clipping range argument if needed.
    """
    if t.dtype in (tf.float32, tf.float64):
      if hasattr(clip, 'dtype'):
        # Convert to tf dtype in case this is a numpy dtype
        clip_dtype = tf.as_dtype(clip.dtype)
        if clip_dtype != t.dtype:
          return tf.cast(clip, t.dtype)
    return clip

  clip_value_min = cast_clip(clip_value_min)
  clip_value_max = cast_clip(clip_value_max)
  return tf.clip_by_value(t, clip_value_min, clip_value_max, name)



def random_exponential(shape, rate=1.0, dtype=tf.float32, seed=None):
  """
  Helper function to sample from the exponential distribution, which is not
  included in core TensorFlow.
  """
  return tf.random_gamma(shape, alpha=1, beta=1. / rate, dtype=dtype, seed=seed)


def random_laplace(shape, loc=0.0, scale=1.0, dtype=tf.float32, seed=None):
  """
  Helper function to sample from the Laplace distribution, which is not
  included in core TensorFlow.
  """
  z1 = random_exponential(shape, loc, dtype=dtype, seed=seed)
  z2 = random_exponential(shape, scale, dtype=dtype, seed=seed)
  return z1 - z2


def random_lp_vector(shape, ord, eps, dtype=tf.float32, seed=None):
  """
  Helper function to generate uniformly random vectors from a norm ball of
  radius epsilon.
  :param shape: Output shape of the random sample. The shape is expected to be
                of the form `(n, d1, d2, ..., dn)` where `n` is the number of
                i.i.d. samples that will be drawn from a norm ball of dimension
                `d1*d1*...*dn`.
  :param ord: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
  :param eps: Epsilon, radius of the norm ball.
  """
  if ord not in [np.inf, 1, 2]:
    raise ValueError('ord must be np.inf, 1, or 2.')

  if ord == np.inf:
    r = tf.random_uniform(shape, -eps, eps, dtype=dtype, seed=seed)
  else:
    dim = tf.reduce_prod(shape[1:])

    if ord == 1:
      x = random_laplace((shape[0], dim), loc=1.0, scale=1.0, dtype=dtype,
                         seed=seed)
      norm = tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True)
    elif ord == 2:
      x = tf.random_normal((shape[0], dim), dtype=dtype, seed=seed)
      norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
    else:
      raise ValueError('ord must be np.inf, 1, or 2.')

    w = tf.pow(tf.random.uniform((shape[0], 1), dtype=dtype, seed=seed),
               1.0 / tf.cast(dim, dtype))
    r = eps * tf.reshape(w * x / norm, shape)
  return r


def clip_eta(eta, ord, eps):
  """
  Helper function to clip the perturbation to epsilon norm ball.
  :param eta: A tensor with the current perturbation.
  :param ord: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
  :param eps: Epsilon, bound of the perturbation.
  """

  # Clipping perturbation eta to ord norm ball
  if ord not in [np.inf, 1, 2]:
    raise ValueError('ord must be np.inf, 1, or 2.')
  reduc_ind = list(range(1, len(eta.get_shape())))
  avoid_zero_div = 1e-12
  if ord == np.inf:
    eta = clip_by_value(eta, -eps, eps)
  elif ord == 1:
    # Implements a projection algorithm onto the l1-ball from
    # (Duchi et al. 2008) that runs in time O(d*log(d)) where d is the
    # input dimension.
    # Paper link (Duchi et al. 2008): https://dl.acm.org/citation.cfm?id=1390191

    eps = tf.cast(eps, eta.dtype)

    dim = tf.reduce_prod(tf.shape(eta)[1:])
    eta_flat = tf.reshape(eta, (-1, dim))
    abs_eta = tf.abs(eta_flat)

    if 'sort' in dir(tf):
      mu = -tf.sort(-abs_eta, axis=-1)
    else:
      # `tf.sort` is only available in TF 1.13 onwards
      mu = tf.nn.top_k(abs_eta, k=dim, sorted=True)[0]
    cumsums = tf.cumsum(mu, axis=-1)
    js = tf.cast(tf.divide(1, tf.range(1, dim + 1)), eta.dtype)
    t = tf.cast(tf.greater(mu - js * (cumsums - eps), 0), eta.dtype)

    rho = tf.argmax(t * cumsums, axis=-1)
    rho_val = tf.reduce_max(t * cumsums, axis=-1)
    theta = tf.divide(rho_val - eps, tf.cast(1 + rho, eta.dtype))

    eta_sgn = tf.sign(eta_flat)
    eta_proj = eta_sgn * tf.maximum(abs_eta - theta[:, tf.newaxis], 0)
    eta_proj = tf.reshape(eta_proj, tf.shape(eta))

    norm = tf.reduce_sum(tf.abs(eta), reduc_ind)
    eta = tf.where(tf.greater(norm, eps), eta_proj, eta)

  elif ord == 2:
    # avoid_zero_div must go inside sqrt to avoid a divide by zero
    # in the gradient through this operation
    norm = tf.sqrt(tf.maximum(avoid_zero_div,
                              reduce_sum(tf.square(eta),
                                         reduc_ind,
                                         keepdims=True)))
    # We must *clip* to within the norm ball, not *normalize* onto the
    # surface of the ball
    factor = tf.minimum(1., div(eps, norm))
    eta = eta * factor
  return eta

def optimize_linear(grad, eps, ord=np.inf):
  """
  Solves for the optimal input to a linear function under a norm constraint.
  Optimal_perturbation = argmax_{eta, ||eta||_{ord} < eps} dot(eta, grad)
  :param grad: tf tensor containing a batch of gradients
  :param eps: float scalar specifying size of constraint region
  :param ord: int specifying order of norm
  :returns:
    tf tensor containing optimal perturbation
  """

  # In Python 2, the `list` call in the following line is redundant / harmless.
  # In Python 3, the `list` call is needed to convert the iterator returned by `range` into a list.
  red_ind = list(range(1, len(grad.get_shape())))
  avoid_zero_div = 1e-12
  if ord == np.inf:
    # Take sign of gradient
    optimal_perturbation = tf.sign(grad)
    # The following line should not change the numerical results.
    # It applies only because `optimal_perturbation` is the output of
    # a `sign` op, which has zero derivative anyway.
    # It should not be applied for the other norms, where the
    # perturbation has a non-zero derivative.
    optimal_perturbation = tf.stop_gradient(optimal_perturbation)
  elif ord == 1:
    abs_grad = tf.abs(grad)
    sign = tf.sign(grad)
    max_abs_grad = tf.reduce_max(abs_grad, red_ind, keepdims=True)
    tied_for_max = tf.to_float(tf.equal(abs_grad, max_abs_grad))
    num_ties = tf.reduce_sum(tied_for_max, red_ind, keepdims=True)
    optimal_perturbation = sign * tied_for_max / num_ties
  elif ord == 2:
    square = tf.maximum(avoid_zero_div,
                        reduce_sum(tf.square(grad),
                                   reduction_indices=red_ind,
                                   keepdims=True))
    optimal_perturbation = grad / tf.sqrt(square)
  else:
    raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                              "currently implemented.")

  # Scale perturbation to be the solution for the norm=eps rather than
  # norm=1 problem
  scaled_perturbation = mul(eps, optimal_perturbation)
  return scaled_perturbation


def pgd_generate(x, model, eps=0.3,eps_iter=0.05, nb_iter=10, y=None, ord=np.inf, clip_min=None, clip_max=None, y_target=None, 
                 rand_init= True, rand_init_eps= 0.3, clip_grad=False, sanity_checks=True):
    """
    Generate symbolic graph for adversarial examples and return.
    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """  
    
    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
      asserts.append(tf.assert_greater_equal(x,
                                             tf.cast(clip_min,
                                                           x.dtype)))
    if clip_max is not None:
      asserts.append(tf.assert_less_equal(x,
                                          tf.cast(clip_max,
                                                        x.dtype)))
    # Initialize loop variables
    if rand_init:
      eta = random_lp_vector(tf.shape(x), ord,
                             tf.cast(rand_init_eps, x.dtype),
                             dtype=x.dtype)
    else:
      eta = tf.zeros(tf.shape(x))

    # Clip eta
    eta = clip_eta(eta, ord, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
      adv_x = clip_by_value(adv_x, clip_min, clip_max)

    if y_target is not None:
      y = y_target
      targeted = True
    elif y is not None:
      y = y
      targeted = False
    else:
      model_preds = model(x)
      preds_max = tf.reduce_max(model_preds, 1, keepdims=True)
      y = tf.to_float(tf.equal(model_preds, preds_max))
      y = tf.stop_gradient(y)
      targeted = False
      del model_preds

#    def cond(i, _):
#      """Iterate until requested number of iterations is completed"""
#      return tf.less(i, nb_iter)
#
#    def body(i, adv_x):
#      """Do a projected gradient step"""
#      adv_x = fgsm_generate(adv_x, model, y=y, eps=eps,  ord=ord, clip_min=clip_min, clip_max=clip_max, 
#                            clip_grad=clip_grad, targeted=targeted, sanity_checks=True)
#     
#      # Clipping perturbation eta to ord norm ball
#      eta = adv_x - x
#      eta = clip_eta(eta, ord, eps)
#      adv_x = x + eta
#
#      # Redo the clipping.
#      # FGM already did it, but subtracting and re-adding eta can add some
#      # small numerical error.
#      if clip_min is not None or clip_max is not None:
#        adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)
#
#      return i + 1, adv_x
#
#    _, adv_x = tf.while_loop(cond, body, (tf.zeros([]), adv_x), back_prop=True,
#                             maximum_iterations=nb_iter)

    for i in range(nb_iter):

        adv_x = fgsm_generate(adv_x, model, y=y, eps=eps_iter,  ord=ord, clip_min=clip_min, clip_max=clip_max, 
                              clip_grad=clip_grad, targeted=targeted, sanity_checks=True)
        #Clipping perturbation eta to ord norm ball
        eta = adv_x - x
        eta = clip_eta(eta, ord, eps)
        adv_x = x + eta
        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = clip_by_value(adv_x, clip_min, clip_max)
        
    common_dtype = tf.float32
    asserts.append(tf.assert_less_equal(tf.cast(eps_iter, dtype=common_dtype), tf.cast(eps, dtype=common_dtype)))
    if ord == np.inf and clip_min is not None:
      asserts.append(tf.assert_less_equal(tf.cast(eps, x.dtype), 1e-6 + tf.cast(clip_max, x.dtype) - tf.cast(clip_min, x.dtype)))
    if sanity_checks:
      with tf.control_dependencies(asserts):
        adv_x = tf.identity(adv_x)

    return adv_x




def fgsm_generate(x, model, y=None, eps=0.3, ord=np.inf, clip_min=None, clip_max=None, clip_grad=False, targeted=False, sanity_checks=True):

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    asserts.append(tf.assert_greater_equal(x, tf.cast(clip_min, x.dtype)))

  if clip_max is not None:
    asserts.append(tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

  logits = model(x)._op.inputs[0]

  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    preds_max = reduce_max(logits, 1, keepdims=True)
    y = tf.to_float(tf.equal(logits, preds_max))
    y = tf.stop_gradient(y)
  y = y / reduce_sum(y, 1, keepdims=True)

  # Compute loss
  
  #################
  ##   CE-loss  ###
  #################
  loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
  if targeted:
    loss = -loss
  
#  ##################
#  ###   CW-loss  ###
#  ##################  
#  logits_sort = tf.contrib.framework.sort(logits, axis=1, direction="DESCENDING")
#  logits_max = tf.gather(logits_sort, axis=1, indices=[0])
#  logits_secondmax = tf.gather(logits_sort, axis=1, indices=[1])  
#  
#  logits_loss = logits_max - logits_secondmax
#  loss = -tf.reduce_mean(logits_loss)
#  if targeted:
#      loss = -loss
      
#  ##################
#  ###   DLR-loss  ###
#  ################## 
#  logits_sort = tf.contrib.framework.sort(logits, axis=1, direction="DESCENDING")
#  logits_max = tf.gather(logits_sort, axis=1, indices=[0])
#  logits_secondmax = tf.gather(logits_sort, axis=1, indices=[1])  
#  logits_thirdmax = tf.gather(logits_sort, axis=1, indices=[2])  
#  
#  logits_loss = tf.divide(logits_max - logits_secondmax, logits_max - logits_thirdmax + 1e12)
#  
#  loss = -tf.reduce_mean(logits_loss)
#  if targeted:
#      loss = -loss
      
  # Define gradient of loss wrt input
  grad, = tf.gradients(loss, x)

  if clip_grad:
    grad = zero_out_clipped_grads(grad, x, clip_min, clip_max)

  optimal_perturbation = optimize_linear(grad, eps, ord)

  # Add perturbation to original example to obtain adversarial example
  adv_x = x + optimal_perturbation

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    # We don't currently support one-sided clipping
    assert clip_min is not None and clip_max is not None
    adv_x = clip_by_value(adv_x, clip_min, clip_max)

  if sanity_checks:
    with tf.control_dependencies(asserts):
      adv_x = tf.identity(adv_x)

  return adv_x









