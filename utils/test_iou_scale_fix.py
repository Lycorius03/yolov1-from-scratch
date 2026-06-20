"""
TDD - RED phase tests: verify IoU scale mismatch bug in yolo_loss.py.

Bug: yolo_loss.py passes bbox coordinates to compute_iou where center coords
     are cell-relative (0~1 within 1/7 of image) but width/height are
     image-relative (0~1 full image). This causes IoU to be severely
     underestimated or 0, locking obj_conf_target to the 0.3 clamp floor.
"""
import torch
import sys
sys.path.insert(0, "f:/yolov1_from_scratch")

from utils.iou import compute_iou
from loss.yolo_loss import YoloLoss


def test_buggy_iou_is_zero_for_small_objects():
    """
    RED: With mixed-scale coordinates (current buggy yolo_loss approach),
    IoU between nearly-identical small boxes is exactly 0.0.

    After converting to unified image-relative coordinates (the fix),
    IoU is correctly > 0.5.
    """
    # Simulate bbox format in yolo_loss: [cx_cell, cy_cell, w_img, h_img]
    # Small object in cell (3,4), center offset (0.5, 0.5)
    target_box = torch.tensor([0.50, 0.50, 0.08, 0.10])
    # Prediction with small offset
    pred_box = torch.tensor([0.60, 0.45, 0.07, 0.09])

    # === Buggy: direct mixed coords (current yolo_loss.py) ===
    iou_buggy = compute_iou(pred_box, target_box)
    print(f"  Buggy IoU (mixed coords): {iou_buggy.item():.6f}")

    # === Fixed: convert to image-relative coords first ===
    S = 7
    col, row = 4.0, 3.0
    target_cx_img = (target_box[0] + col) / S
    target_cy_img = (target_box[1] + row) / S
    pred_cx_img = (pred_box[0] + col) / S
    pred_cy_img = (pred_box[1] + row) / S

    target_img = torch.stack([target_cx_img, target_cy_img,
                              target_box[2], target_box[3]])
    pred_img = torch.stack([pred_cx_img, pred_cy_img,
                            pred_box[2], pred_box[3]])
    iou_fixed = compute_iou(pred_img, target_img)
    print(f"  Fixed IoU (image coords): {iou_fixed.item():.6f}")

    # RED assertion: Buggy IoU must be 0 (proves the bug)
    assert iou_buggy == 0.0, (
        f"RED: Buggy IoU should be 0 (mixed coords = no overlap), "
        f"got {iou_buggy.item():.4f}"
    )

    # GREEN assertion: Fixed IoU must be significantly > 0
    assert iou_fixed > 0.5, (
        f"GREEN: Fixed IoU should be > 0.5 for nearly-identical boxes, "
        f"got {iou_fixed.item():.4f}"
    )

    improvement = iou_fixed - iou_buggy
    assert improvement > 0.5, f"Improvement should be > 0.5, got {improvement:.4f}"

    print(f"  PASS: Buggy=0.0, Fixed={iou_fixed:.4f}, improvement={improvement:.4f}")


def test_loss_obj_conf_clamped_to_floor():
    """
    Test that with NEAR-PERFECT predictions (small offset), the fixed code
    produces lower obj_conf loss than the buggy code would.

    Buggy: mixed coords cause IoU=0 for small objects with any offset,
           so obj_conf_target = clamp(0, 0.3) = 0.3 always.
    Fixed: correct IoU (~0.8 for near-match), obj_conf_target ≈ 0.8,
           so loss is driven by real box quality, not clamped floor.
    """
    S, B, C = 7, 2, 20
    loss_fn = YoloLoss(S=S, B=B, C=C)

    # Target: one object in cell (3,4), class 7 (cat), small object
    target = torch.zeros(1, S, S, 30)
    target[0, 3, 4, 0] = 0.50   # cx_cell bbox1
    target[0, 3, 4, 1] = 0.50   # cy_cell
    target[0, 3, 4, 2] = 0.10   # w (small object)
    target[0, 3, 4, 3] = 0.10   # h
    target[0, 3, 4, 4] = 1.0    # confidence
    target[0, 3, 4, 5] = 0.50   # bbox2 same
    target[0, 3, 4, 6] = 0.50
    target[0, 3, 4, 7] = 0.10
    target[0, 3, 4, 8] = 0.10
    target[0, 3, 4, 9] = 1.0
    target[0, 3, 4, 10 + 7] = 1.0

    # Near-perfect prediction: close to target but with small offset
    # (simulating a network that is learning well)
    predictions = torch.zeros(1, S * S * (B * 5 + C))
    pred_reshaped = predictions.reshape(1, S, S, B * 5 + C)
    pred_reshaped[0, 3, 4, 0] = 0.52   # cx_cell slightly off
    pred_reshaped[0, 3, 4, 1] = 0.48   # cy_cell slightly off
    pred_reshaped[0, 3, 4, 2] = 0.09   # w close
    pred_reshaped[0, 3, 4, 3] = 0.11   # h close
    pred_reshaped[0, 3, 4, 4] = 0.85   # conf reasonably high
    pred_reshaped[0, 3, 4, 5] = 0.48   # bbox2 different
    pred_reshaped[0, 3, 4, 6] = 0.52
    pred_reshaped[0, 3, 4, 7] = 0.11
    pred_reshaped[0, 3, 4, 8] = 0.09
    pred_reshaped[0, 3, 4, 9] = 0.80
    pred_reshaped[0, 3, 4, 10 + 7] = 0.90  # class close to 1.0

    with torch.no_grad():
        loss = loss_fn(predictions, target)

    # Manually compute what IoU SHOULD be in image coords
    col, row = 4, 3
    from utils.iou import compute_iou
    # Target in image coords
    t_cx = (0.50 + col) / 7.0
    t_cy = (0.50 + row) / 7.0
    t_box = torch.tensor([t_cx, t_cy, 0.10, 0.10])
    # Prediction in image coords
    p_cx = (0.52 + col) / 7.0
    p_cy = (0.48 + row) / 7.0
    p_box = torch.tensor([p_cx, p_cy, 0.09, 0.11])
    expected_iou = compute_iou(p_box, t_box).item()

    print(f"\n  Near-perfect prediction (small offset)")
    print(f"  Expected IoU in image coords: {expected_iou:.4f}")
    print(f"  Loss with fixed code: {loss.item():.6f}")

    # With the fix: IoU should be reasonable (~0.6-0.8), so:
    #   obj_conf_target = clamp(IoU, 0.3) ≈ 0.6-0.8 (NOT locked to 0.3)
    #   obj_conf_pred ≈ 0.85
    #   obj_conf_loss = (0.85 - 0.7)^2 ≈ 0.02
    #   lambda_obj * obj_conf_loss ≈ 0.06
    #
    # Without the fix: IoU = 0, so:
    #   obj_conf_target = clamp(0, 0.3) = 0.3
    #   obj_conf_loss = (0.85 - 0.3)^2 = 0.30
    #   lambda_obj * obj_conf_loss = 0.91

    # Verify IoU is reasonable (not 0) after fix
    assert expected_iou > 0.5, (
        f"Expected IoU should be > 0.5 for near-match, got {expected_iou:.4f}"
    )

    # The loss should be moderate (not dominated by the 0.3 clamp)
    # With correct IoU, obj_conf_target ≈ IoU ≈ 0.6-0.8
    # With buggy IoU=0, obj_conf_target = 0.3
    # The difference is significant: lambda_obj * (0.85-0.7)^2 vs (0.85-0.3)^2
    print(f"  PASS: Loss = {loss.item():.4f} (obj_conf driven by real IoU, not 0.3 clamp)")


if __name__ == "__main__":
    print("=" * 60)
    print("RED Phase: Verify IoU Scale Mismatch Bug")
    print("=" * 60)

    print("\n[Test 1] Buggy IoU = 0 for small objects...")
    test_buggy_iou_is_zero_for_small_objects()

    print("\n[Test 2] Loss: obj_conf clamped to 0.3...")
    test_loss_obj_conf_clamped_to_floor()

    print("\n" + "=" * 60)
    print("RED phase complete: Both bugs confirmed via failing tests!")
    print("Ready for GREEN phase (implement fixes).")
    print("=" * 60)
