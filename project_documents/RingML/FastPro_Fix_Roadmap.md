# FastPro Remediation Roadmap

**Date:** December 11, 2025
**Author:** Azzeddine Remmal
**Status:** Completed

## 1. Vision
To restore the performance and stability of the RingML library by fixing critical C-level bugs in the `FastPro` extension. This will allow RingML to switch from slow "Safe Mode" (Ring loops) back to high-speed C operations.

## 2. Functional Modules

### Module 1: Core Precision Fixes
**Goal:** Ensure floating-point precision is preserved in all scalar operations.
**Rationale:** Deep learning requires precise gradients (e.g., 0.001). Casting to `int` destroys this.

### Module 2: Stability Fixes
**Goal:** Prevent crashes in Matrix operations and ensure consistent API behavior.
**Rationale:** `transpose` crashes on non-square matrices. In-place operations must return the list reference to support method chaining.

### Module 3: Feature Expansion
**Goal:** Support Element-Wise Multiplication.
**Rationale:** Essential for calculating gradients (Hadamard Product).

## 3. Detailed To-Do List

### Module 1: Core Precision Fixes
- [x] **T01.01**: Remove `(int)` cast in `ring_updatelist` parameter parsing (Line ~565).
  - *File:* `fastpro.c`
  - *Action:* Change `nValue = (int) RING_API_GETNUMBER(4)` to `nValue = (double) RING_API_GETNUMBER(4)`.

### Module 2: Stability Fixes
- [x] **T02.01**: Fix Transpose Logic (Case 1406).
  - *File:* `fastpro.c`
  - *Action:* Correct the nested loop indexing to properly handle non-square matrices ($N \times M \to M \times N$).
- [x] **T02.02**: Fix Return Values for In-Place Operations.
  - *File:* `fastpro.c`
  - *Action:* Add `RING_API_RETLIST(pList)` to Case 1906 (`identity`) and Case 2006 (`random`).

### Module 3: Feature Expansion
- [x] **T03.01**: Implement Element-Wise Multiplication.
  - *File:* `fastpro.c`
  - *Action:* 
    1. Add `emul` string check in parsing section (New OpCode range, e.g., 4500).
    2. Implement Case 4506 (Element-Wise Matrix Mul).

### Module 4: New Features
- [x] **T04.01**: Softmax Implementation (Case 3306).
  - *File:* `fastpro.c`
  - *Action:* Implement Case 3306 in C.

