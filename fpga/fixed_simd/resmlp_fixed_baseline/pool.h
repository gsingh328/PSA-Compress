#ifndef POOL_H
#define POOL_H

#include "accel_common.h"


void avgpool(
	accel_io_t io_buff[IO_BUFF_N][IO_BUFF_DIM_0][IO_BUFF_DIM_1],
	const dim_t io_x_idx,
	const dim_t io_y_idx,
	const dim_t x_dim_0,
	const dim_t x_dim_1,
	const dim_t x_dim_1_lg_2
);


#endif // POOL_H
