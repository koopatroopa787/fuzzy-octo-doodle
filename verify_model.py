"""
Quick verification script to check model architecture and parameter count.
Run this to verify the model meets requirements without training.
"""
import torch
from submission.fashion_model import Net


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_model():
    """Verify model architecture and requirements."""
    print("="*60)
    print("MODEL VERIFICATION")
    print("="*60)

    # Initialize model
    model = Net()

    # Count parameters
    total_params = count_parameters(model)

    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(dummy_input)
        forward_pass_ok = output.shape == (1, 10)
    except Exception as e:
        print(f"ERROR: Forward pass failed: {e}")
        return

    # Print architecture summary
    print("\nModel Architecture:")
    print(model)

    print(f"\n{'='*60}")
    print("PARAMETER COUNT:")
    print(f"{'='*60}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:30s}: {param.numel():>10,} params")

    print(f"{'='*60}")
    print(f"{'Total Parameters:':<30s}  {total_params:>10,}")
    print(f"{'Parameter Limit:':<30s}  {100000:>10,}")
    print(f"{'Usage:':<30s}  {100 * total_params / 100000:>9.1f}%")
    print(f"{'='*60}")

    # Check requirements
    print("\nREQUIREMENT CHECKS:")
    print(f"{'='*60}")

    if total_params <= 100000:
        print("✓ PASS: Parameter count ≤ 100,000")
        if total_params <= 70000:
            print("✓ BONUS: Very efficient (<70K params - likely bottom 30th percentile)")
        elif total_params <= 85000:
            print("  INFO: Good efficiency (<85K params)")
    else:
        print("✗ FAIL: Exceeds parameter limit!")

    if forward_pass_ok:
        print("✓ PASS: Forward pass works correctly")
        print(f"  Output shape: {output.shape} (expected: (1, 10))")
    else:
        print("✗ FAIL: Forward pass failed!")

    print(f"{'='*60}")

    # Expected accuracy range
    print("\nEXPECTED PERFORMANCE:")
    print(f"{'='*60}")
    print("With optimized training (SGD + Cosine LR + Label Smoothing):")
    print("  • Estimated accuracy: 92-94% on test set")
    print("  • Training time: ~10-15 minutes on CPU, ~2-3 minutes on GPU")
    print("  • Convergence: ~40-50 epochs with early stopping")
    print(f"{'='*60}")

    print("\nTRAINING INSTRUCTIONS:")
    print("  To train the model, run:")
    print("    uv run python -m submission.fashion_training")
    print()
    print("  The trained weights will be saved to:")
    print("    submission/model_weights.pth")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    verify_model()
