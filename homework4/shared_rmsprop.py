from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python.training.optimizer import Optimizer
from tensorflow import group, assign, control_dependencies


class SharedRMSPropOptimizer(RMSPropOptimizer):
  def minimize(self, loss, var_list, global_var_list, global_step=None,
               gate_gradients=Optimizer.GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None, sync_locals_after_update=True):
    """Add operations to minimize `loss` by updating and reading global vars
    """

    if global_var_list is None or not global_var_list:
      raise ValueError(
        "No global variables provided. Using regular RMSPropOptimizer"
        " is recommended when no global variables need to be updated")

    grads_and_vars = self.compute_gradients(
      loss, var_list=var_list, gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops,
      grad_loss=grad_loss)

    grads, local_vars = list(zip(*grads_and_vars))
    grads_and_global_vars = list(zip(grads, global_var_list))

    vars_with_grad = [v for g, v in grads_and_global_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
          "No gradients provided for any variable, check your graph for ops"
          " that do not support gradients, between variables %s and loss %s." %
          ([str(v) for _, v in grads_and_vars], loss))

    apply_gradients = self.apply_gradients(grads_and_global_vars,
                                           global_step=global_step,
                                           name=name)

    if not sync_locals_after_update:
      return apply_gradients

    with control_dependencies([apply_gradients]):
      update_local_vars = group(*[
        assign(local_var, global_var)
        for global_var, local_var in
        zip(global_var_list, var_list)
      ])

    return group(apply_gradients, update_local_vars)
