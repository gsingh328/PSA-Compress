#ifndef UTILS_H
#define UTILS_H

// Helper Defines

#define IDX2D(i,j,N) (((i)*(N)) + (j))

#define TO_BUFF(arr) (&arr[0])
#define TO_BUFF2(arr) (&arr[0][0])

// Macro to cast an n-dimensional array to a 1-dimensional array
#define CAST_TO_1D_ARRAY(array, type) ((type*)(array))

#define CLIP(x, MIN, MAX) {				\
	if (x > MAX) {						\
		x = MAX;						\
	} else if (x < MIN) {				\
		x = MIN;						\
	} else	{							\
		x = x;							\
	}									\
}

#define CLIP_MAX(x, MAX) x = (x > MAX) ? (MAX) : (x)

// if(s<=1) {std::cout << "NEGATIVE! = " << s << "\n"; return;}\
// if(s>=16) {std::cout << "LARGE! = " << s << "\n"; return;}\

// #define ROUNDED_SHIFT(x, s) {			\
// 	if (s > 0 && s < 32) { 				\
// 		ap_uint<1> rnd_factor = x[s-1];	\
// 		x = (x >> s) + rnd_factor;		\
// 	} 									\
// }

#define ROUNDED_SHIFT(x, s) \
x = (x >> s) + x[s-1];

// #define ROUNDED_SHIFT(x, s) {		\
// 	if (s > 0) { 					\
// 		x >>= (s - 1);				\
// 		x += (x & 1);				\
// 		x >>= 1;					\
// 	} 								\
// }


#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)

#define DIV_ROUND(n, d) ((n + (d >> 1)) / d)

#define CEIL_MULTIPLE(n, d) (DIV_ROUNDUP(n, d) * d)

#endif // UTILS_H
