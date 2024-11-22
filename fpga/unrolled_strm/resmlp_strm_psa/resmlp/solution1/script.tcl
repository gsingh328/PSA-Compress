############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
## Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
############################################################
open_project resmlp
set_top krnl_resmlp
add_files quant_params.h
add_files resmlp.cpp
add_files resmlp.h
add_files resmlp_block_macros.h
add_files resmlp_blocks.h
add_files resmlp_extra_layers.h
add_files resmlp_mlp.h
add_files resmlp_model.h
add_files resmlp_token_mixer.h
add_files resmlp_types.h
add_files resmlp_utils.h
add_files -tb cifar10_tb.cpp -cflags "-Wno-unknown-pragmas"
add_files -tb quant_samples.h -cflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vitis
set_part {xcu50-fsvh2104-2-e}
create_clock -period 5 -name default
config_cosim -tool xsim
config_export -flow impl -format ip_catalog -rtl verilog -vivado_clock 5
config_interface -m_axi_alignment_byte_size 64 -m_axi_latency 64 -m_axi_max_widen_bitwidth 512
config_rtl -register_reset_num 3
source "./resmlp/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -flow impl -rtl verilog -format ip_catalog
