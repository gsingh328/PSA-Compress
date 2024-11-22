#ifndef RESMLP_UTILS_H
#define RESMLP_UTILS_H


#define CLIP(x, MIN, MAX)\
x = (x > MAX) ? (MAX) : ((x < MIN) ? (MIN) : x)

// #define CLIP(x, MIN, MAX)\
// if (x < MIN) x = MIN; \
// else if (x > MAX) x = MAX; \
// else x = x;

#define ROUNDED_SHIFT(x, s) \
x = (x >> s) + x[s-1];

// #define ROUNDED_SHIFT(x, s) \
// x = (x >> s) + ((x >> (s-1)) & 1);

// #define ROUNDED_SHIFT(x, s) \
// x = x >> (s-1); \
// ap_uint<1> rnd_factor = x & 1; \
// x = (x >> 1) + rnd_factor;


#endif // RESMLP_UTILS_H
