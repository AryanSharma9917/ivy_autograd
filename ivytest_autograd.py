import ivy
from ivy.core.autodiff import GraphManagers
from ivy_tests import helpers
from ivy.core.container import Container

def test_ivy_autograd_tan(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    gm = GraphManagers.select_graph_manager(x[0])
    with gm:
        ivy_x = ivy.Variable(x[0], requires_grad=True)
        ivy_out = ivy.tan(ivy_x)
        assert ivy_out.shape == x[0].shape
        ivy_grad_out = ivy.ones_like(ivy_out)
        ivy_out.backward(ivy_grad_out)
        np_out_grad = ivy.to_numpy(ivy_x.grad)
    if not as_variable:
        np_out = ivy.to_numpy(ivy_out)
        np_x = ivy.to_numpy(ivy_x)
    else:
        np_out = ivy_out
        np_x = ivy_x
    container = Container(
        {'input': np_x}, 
        input_grads={'input': np_out_grad},
        output_grads={'output': np_out_grad}
    )
    helpers.assert_close(fn_tree=fn_tree,
                         native_array_flags=native_array,
                         on_device=on_device,
                         container=container,
                         frontend=frontend,
                         **as_variable,
                         **with_out)
      
      
# This function creates an autograd tape using the GraphManagers API from Ivy, and then applies the ivy.tan operation to the input tensor ivy_x. 
#It computes the gradients of the output tensor with respect to the input tensor using the autograd machinery from Ivy. Finally, it constructs a Container object 
#with the input tensor and output gradients, and calls the assert_close function from the ivy_tests.helpers module to test the output of the function.
