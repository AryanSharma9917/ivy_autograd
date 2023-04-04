import torch
import ivy

class AddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other, *, alpha=None, out=None):
        input_ivy, other_ivy = ivy.to_ivy(input), ivy.to_ivy(other)
        result_ivy = ivy.add(input_ivy, other_ivy, alpha=alpha, out=out)
        result = ivy.to_numpy(result_ivy)
        if out is not None:
            out.copy_(result)
            result = out
        ctx.save_for_backward(input, other, alpha)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, other, alpha = ctx.saved_tensors
        grad_input_ivy = ivy.to_ivy(grad_output)
        if input.requires_grad:
            grad_input_ivy *= 1.0
        grad_input = ivy.to_numpy(grad_input_ivy)
        grad_other_ivy = ivy.to_ivy(grad_output)
        if other.requires_grad:
            if alpha is not None:
                grad_other_ivy *= alpha
            else:
                grad_other_ivy *= 1.0
        grad_other = ivy.to_numpy(grad_other_ivy)
        return grad_input, grad_other, None, None
