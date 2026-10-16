/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef WASM_CANONICAL_NAN_H
#define WASM_CANONICAL_NAN_H

/* Arithmetic results only. Bitwise operations and transport preserve payloads. */
static inline float
wasm_canonical_F32(float value)
{
    union { float value; uint32 bits; } result = { .value = value };
    if ((result.bits & 0x7fffffffU) > 0x7f800000U)
        result.bits = 0x7fc00000U;
    return result.value;
}

static inline double
wasm_canonical_F64(double value)
{
    union { double value; uint64 bits; } result = { .value = value };
    if ((result.bits & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL)
        result.bits = 0x7ff8000000000000ULL;
    return result.value;
}
#endif
