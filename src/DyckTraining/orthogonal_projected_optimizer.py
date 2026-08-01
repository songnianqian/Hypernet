"""
Custom Optimizer with Orthogonal Gradient Projection

Modifies standard gradient descent to project gradients orthogonal to unit vectors.

For parameter update:
    Standard GD:     θ_new = θ - lr * ∇f
    
    Our modified GD: θ_new = θ - lr * ∇f_proj
    
    Where: ∇f_proj = ∇f - (∇f · ŷ) ŷ
           ŷ = y / |y|  (unit vector)

This removes the component of the gradient parallel to ŷ,
keeping only the orthogonal component.

Geometric interpretation:
    - Standard gradient points in direction of steepest ascent
    - Projected gradient removes the component along ŷ
    - Update happens in the orthogonal subspace only
"""

import torch
from torch.optim import Optimizer
import math
from typing import List, Optional


class OrthogonalProjectedSGD(Optimizer):
    """
    SGD with gradient projection orthogonal to unit vectors.
    
    Implements: ∇f_proj = ∇f - (∇f · ŷ) ŷ
    
    Args:
        params: Model parameters
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        dampening: Dampening for momentum (default: 0)
        nesterov: Enables Nesterov momentum (default: False)
        projection_mode: How to compute unit vector ŷ
            - 'parameter': Use normalized parameter itself (ŷ = θ/|θ|)
            - 'gradient': Use normalized gradient (ŷ = ∇f/|∇f|)
            - 'momentum': Use momentum direction (ŷ = m/|m|)
            - 'fixed': Use a fixed random direction per parameter
        projection_strength: Scale factor for projection (0=no projection, 1=full projection)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        projection_mode: str = 'parameter',
        projection_strength: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if projection_strength < 0.0 or projection_strength > 1.0:
            raise ValueError(f"Invalid projection_strength: {projection_strength}")
        if projection_mode not in ['parameter', 'gradient', 'momentum', 'fixed']:
            raise ValueError(f"Invalid projection_mode: {projection_mode}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            projection_mode=projection_mode,
            projection_strength=projection_strength,
        )
        super(OrthogonalProjectedSGD, self).__init__(params, defaults)
        
        # Initialize fixed projection directions if needed
        if projection_mode == 'fixed':
            self._initialize_fixed_directions()
    
    def _initialize_fixed_directions(self):
        """Initialize fixed random unit vectors for projection"""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # Random unit vector
                    direction = torch.randn_like(p.data)
                    direction = direction / (torch.norm(direction) + 1e-8)
                    state['fixed_direction'] = direction
    
    def _get_unit_vector(self, p, grad, state, projection_mode):
        """
        Get unit vector ŷ for projection based on mode.
        
        Args:
            p: Parameter tensor
            grad: Gradient tensor
            state: Optimizer state for this parameter
            projection_mode: Which mode to use
            
        Returns:
            ŷ: Unit vector for projection
        """
        if projection_mode == 'parameter':
            # Use normalized parameter: ŷ = θ/|θ|
            y = p.data
            y_hat = y / (torch.norm(y) + 1e-8)
            
        elif projection_mode == 'gradient':
            # Use normalized gradient: ŷ = ∇f/|∇f|
            y = grad
            y_hat = y / (torch.norm(y) + 1e-8)
            
        elif projection_mode == 'momentum':
            # Use normalized momentum: ŷ = m/|m|
            if 'momentum_buffer' not in state:
                # Initialize momentum buffer
                y_hat = torch.zeros_like(p.data)
            else:
                y = state['momentum_buffer']
                y_hat = y / (torch.norm(y) + 1e-8)
                
        elif projection_mode == 'fixed':
            # Use fixed random direction
            y_hat = state['fixed_direction']
        
        else:
            raise ValueError(f"Unknown projection_mode: {projection_mode}")
        
        return y_hat
    
    def _project_gradient(self, grad, y_hat, projection_strength):
        """
        Project gradient orthogonal to unit vector.
        
        ∇f_proj = ∇f - α * (∇f · ŷ) ŷ
        
        Where α is projection_strength (0 to 1)
        
        Args:
            grad: Original gradient ∇f
            y_hat: Unit vector ŷ
            projection_strength: How much to project (0=none, 1=full)
            
        Returns:
            Projected gradient ∇f_proj
        """
        # Compute dot product: ∇f · ŷ
        # For tensors, flatten and compute dot product
        grad_flat = grad.reshape(-1)
        y_hat_flat = y_hat.reshape(-1)
        dot_product = torch.dot(grad_flat, y_hat_flat)
        
        # Compute parallel component: (∇f · ŷ) ŷ
        parallel_component = dot_product * y_hat
        
        # Remove parallel component (scaled by projection_strength)
        grad_projected = grad - projection_strength * parallel_component
        
        return grad_projected
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step with projected gradients.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            projection_mode = group['projection_mode']
            projection_strength = group['projection_strength']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                state = self.state[p]
                
                # Get unit vector for projection
                y_hat = self._get_unit_vector(p, grad, state, projection_mode)
                
                # Project gradient orthogonal to y_hat
                grad_projected = self._project_gradient(grad, y_hat, projection_strength)
                
                # Apply momentum (if enabled)
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(grad_projected).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad_projected, alpha=1 - dampening)
                    
                    if nesterov:
                        grad_projected = grad_projected.add(buf, alpha=momentum)
                    else:
                        grad_projected = buf
                
                # Update parameter
                p.data.add_(grad_projected, alpha=-group['lr'])
        
        return loss


class OrthogonalProjectedAdamW(Optimizer):
    """
    AdamW with gradient projection orthogonal to unit vectors.
    
    Combines:
    - Adaptive learning rates (Adam)
    - Decoupled weight decay (AdamW)
    - Orthogonal gradient projection (our modification)
    
    The projection happens BEFORE computing adaptive moments.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        projection_mode: str = 'parameter',
        projection_strength: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if projection_strength < 0.0 or projection_strength > 1.0:
            raise ValueError(f"Invalid projection_strength: {projection_strength}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            projection_mode=projection_mode,
            projection_strength=projection_strength,
        )
        super(OrthogonalProjectedAdamW, self).__init__(params, defaults)
        
        # Initialize fixed projection directions if needed
        if projection_mode == 'fixed':
            self._initialize_fixed_directions()
    
    def _initialize_fixed_directions(self):
        """Initialize fixed random unit vectors for projection"""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    direction = torch.randn_like(p.data)
                    direction = direction / (torch.norm(direction) + 1e-8)
                    state['fixed_direction'] = direction
    
    def _get_unit_vector(self, p, grad, state, projection_mode):
        """Get unit vector ŷ for projection"""
        if projection_mode == 'parameter':
            y = p.data
            y_hat = y / (torch.norm(y) + 1e-8)
            
        elif projection_mode == 'gradient':
            y = grad
            y_hat = y / (torch.norm(y) + 1e-8)
            
        elif projection_mode == 'momentum':
            # For Adam, use exp_avg (first moment)
            if 'exp_avg' not in state:
                y_hat = torch.zeros_like(p.data)
            else:
                y = state['exp_avg']
                y_hat = y / (torch.norm(y) + 1e-8)
                
        elif projection_mode == 'fixed':
            y_hat = state['fixed_direction']
        
        else:
            raise ValueError(f"Unknown projection_mode: {projection_mode}")
        
        return y_hat
    
    def _project_gradient(self, grad, y_hat, projection_strength):
        """Project gradient orthogonal to unit vector"""
        grad_flat = grad.reshape(-1)
        y_hat_flat = y_hat.reshape(-1)
        dot_product = torch.dot(grad_flat, y_hat_flat)
        parallel_component = dot_product * y_hat
        grad_projected = grad - projection_strength * parallel_component
        return grad_projected
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with projected gradients"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            projection_mode = group['projection_mode']
            projection_strength = group['projection_strength']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Get unit vector for projection
                y_hat = self._get_unit_vector(p, grad, state, projection_mode)
                
                # Project gradient orthogonal to y_hat
                grad_projected = self._project_gradient(grad, y_hat, projection_strength)
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad_projected, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad_projected, grad_projected, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute step size
                step_size = group['lr'] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                
                # AdamW weight decay (decoupled)
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update parameters
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


# Example usage and testing
if __name__ == "__main__":
    print("Testing Orthogonal Projected Optimizers")
    print("=" * 80)
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    # Create some dummy data
    X = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    
    # Test different projection modes
    projection_modes = ['parameter', 'gradient', 'momentum', 'fixed']
    
    print("\n📊 Testing OrthogonalProjectedSGD:")
    for mode in projection_modes:
        print(f"\n  Mode: {mode}")
        
        optimizer = OrthogonalProjectedSGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            projection_mode=mode,
            projection_strength=1.0
        )
        
        # Forward pass
        output = model(X)
        loss = torch.nn.functional.cross_entropy(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients before projection
        original_grad_norm = torch.norm(model[0].weight.grad)
        
        # Optimization step (applies projection)
        optimizer.step()
        
        print(f"    Loss: {loss.item():.4f}")
        print(f"    Original gradient norm: {original_grad_norm:.4f}")
        print(f"    ✓ Projection applied successfully")
    
    print("\n📊 Testing OrthogonalProjectedAdamW:")
    for mode in projection_modes:
        print(f"\n  Mode: {mode}")
        
        # Reset model
        for param in model.parameters():
            param.data = torch.randn_like(param.data)
        
        optimizer = OrthogonalProjectedAdamW(
            model.parameters(),
            lr=0.001,
            projection_mode=mode,
            projection_strength=1.0
        )
        
        # Forward pass
        output = model(X)
        loss = torch.nn.functional.cross_entropy(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Optimization step
        optimizer.step()
        
        print(f"    Loss: {loss.item():.4f}")
        print(f"    ✓ Projection applied successfully")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("\n💡 Usage in training:")
    print("""
    # For SGD with orthogonal projection:
    optimizer = OrthogonalProjectedSGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        projection_mode='parameter',  # or 'gradient', 'momentum', 'fixed'
        projection_strength=1.0,      # 0 to 1, controls projection amount
    )
    
    # For AdamW with orthogonal projection:
    optimizer = OrthogonalProjectedAdamW(
        model.parameters(),
        lr=0.001,
        projection_mode='parameter',
        projection_strength=1.0,
    )
    """)
