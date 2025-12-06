/*------------------------------------------------------------------------
 *
 *  Copyright (c) 2021 by Bo Young Kang, All rights reserved.
 *
 *  File name  : conv1_calc.v
 *  Written by : Kang, Bo Young, weenslab
 *  Written on : Oct 1, 2021
 *  Version    : 22
 *  Design     : 1st Convolution Layer for CNN MNIST dataset
 *               Convolution Sum Calculation
 *
 *------------------------------------------------------------------------*/

/*-------------------------------------------------------------------
 *  Module: conv1_calc
 *------------------------------------------------------------------*/

module conv1_calc
    #(
        parameter WIDTH = 28,
                  HEIGHT = 28,
                  DATA_BITS = 8
    )
    (
        input  wire                 clk,
        input  wire                 rst_n,
        input  wire                 valid_out_buf,
        input  wire [DATA_BITS-1:0] data_out_0, data_out_1, data_out_2, data_out_3, data_out_4,
                                    data_out_5, data_out_6, data_out_7, data_out_8, data_out_9,
                                    data_out_10, data_out_11, data_out_12, data_out_13, data_out_14,
                                    data_out_15, data_out_16, data_out_17, data_out_18, data_out_19,
                                    data_out_20, data_out_21, data_out_22, data_out_23, data_out_24,
        output wire signed [11:0]   conv_out_1, conv_out_2, conv_out_3,
        output wire                 valid_out_calc
    );

    localparam FILTER_SIZE = 5;
    localparam CHANNEL_LEN = 3;

    reg signed [DATA_BITS-1:0] weight_1 [0:FILTER_SIZE*FILTER_SIZE-1];
    reg signed [DATA_BITS-1:0] weight_2 [0:FILTER_SIZE*FILTER_SIZE-1];
    reg signed [DATA_BITS-1:0] weight_3 [0:FILTER_SIZE*FILTER_SIZE-1];
    reg signed [DATA_BITS-1:0] bias [0:CHANNEL_LEN-1];

    reg signed [19:0] calc_out_1_tmp0, calc_out_1_tmp1, calc_out_1_tmp2, calc_out_1_tmp3,
                      calc_out_1_tmp4, calc_out_1_tmp5, calc_out_1_tmp6, calc_out_1_tmp7,
                      calc_out_1_tmp8, calc_out_1_tmp9, calc_out_1_tmp10, calc_out_1_tmp11,
                      calc_out_1_tmp12, calc_out_1_tmp13, calc_out_1_tmp14, calc_out_1_tmp15,
                      calc_out_1_tmp16, calc_out_1_tmp17, calc_out_1_tmp18, calc_out_1_tmp19,
                      calc_out_1_tmp20, calc_out_1_tmp21, calc_out_1_tmp22;
    reg signed [19:0] calc_out_2_tmp0, calc_out_2_tmp1, calc_out_2_tmp2, calc_out_2_tmp3,
                      calc_out_2_tmp4, calc_out_2_tmp5, calc_out_2_tmp6, calc_out_2_tmp7,
                      calc_out_2_tmp8, calc_out_2_tmp9, calc_out_2_tmp10, calc_out_2_tmp11,
                      calc_out_2_tmp12, calc_out_2_tmp13, calc_out_2_tmp14, calc_out_2_tmp15,
                      calc_out_2_tmp16, calc_out_2_tmp17, calc_out_2_tmp18, calc_out_2_tmp19,
                      calc_out_2_tmp20, calc_out_2_tmp21, calc_out_2_tmp22;
    reg signed [19:0] calc_out_3_tmp0, calc_out_3_tmp1, calc_out_3_tmp2, calc_out_3_tmp3,
                      calc_out_3_tmp4, calc_out_3_tmp5, calc_out_3_tmp6, calc_out_3_tmp7,
                      calc_out_3_tmp8, calc_out_3_tmp9, calc_out_3_tmp10, calc_out_3_tmp11,
                      calc_out_3_tmp12, calc_out_3_tmp13, calc_out_3_tmp14, calc_out_3_tmp15,
                      calc_out_3_tmp16, calc_out_3_tmp17, calc_out_3_tmp18, calc_out_3_tmp19,
                      calc_out_3_tmp20, calc_out_3_tmp21, calc_out_3_tmp22;

    wire signed [19:0] calc_out_1, calc_out_2, calc_out_3;
    wire signed [DATA_BITS:0] exp_data [0:FILTER_SIZE*FILTER_SIZE-1];
    wire signed [11:0] exp_bias [0:CHANNEL_LEN-1];

    reg valid_out_buf_tmp0, valid_out_buf_tmp1, valid_out_buf_tmp2, valid_out_buf_tmp3;

    initial
    begin
        $readmemh("conv1_weight_1.mem", weight_1);
        $readmemh("conv1_weight_2.mem", weight_2);
        $readmemh("conv1_weight_3.mem", weight_3);
        $readmemh("conv1_bias.mem", bias);
    end

    // Unsigned -> Signed
    assign exp_data[0] = {1'd0, data_out_0};
    assign exp_data[1] = {1'd0, data_out_1};
    assign exp_data[2] = {1'd0, data_out_2};
    assign exp_data[3] = {1'd0, data_out_3};
    assign exp_data[4] = {1'd0, data_out_4};
    assign exp_data[5] = {1'd0, data_out_5};
    assign exp_data[6] = {1'd0, data_out_6};
    assign exp_data[7] = {1'd0, data_out_7};
    assign exp_data[8] = {1'd0, data_out_8};
    assign exp_data[9] = {1'd0, data_out_9};
    assign exp_data[10] = {1'd0, data_out_10};
    assign exp_data[11] = {1'd0, data_out_11};
    assign exp_data[12] = {1'd0, data_out_12};
    assign exp_data[13] = {1'd0, data_out_13};
    assign exp_data[14] = {1'd0, data_out_14};
    assign exp_data[15] = {1'd0, data_out_15};
    assign exp_data[16] = {1'd0, data_out_16};
    assign exp_data[17] = {1'd0, data_out_17};
    assign exp_data[18] = {1'd0, data_out_18};
    assign exp_data[19] = {1'd0, data_out_19};
    assign exp_data[20] = {1'd0, data_out_20};
    assign exp_data[21] = {1'd0, data_out_21};
    assign exp_data[22] = {1'd0, data_out_22};
    assign exp_data[23] = {1'd0, data_out_23};
    assign exp_data[24] = {1'd0, data_out_24};

    //  Re-calibration of extracted weight data according to MSB
    assign exp_bias[0] = (bias[0][7] == 1) ? {4'b1111, bias[0]} : {4'd0, bias[0]};
    assign exp_bias[1] = (bias[1][7] == 1) ? {4'b1111, bias[1]} : {4'd0, bias[1]};
    assign exp_bias[2] = (bias[2][7] == 1) ? {4'b1111, bias[2]} : {4'd0, bias[2]};

    // Multi stage multiplication and addition for out_1
    always @(posedge clk) // Stage 1
    begin
        if (~rst_n)
        begin
            calc_out_1_tmp0 <= 0;
            calc_out_1_tmp1 <= 0;
            calc_out_1_tmp2 <= 0;
            calc_out_1_tmp3 <= 0;
            calc_out_1_tmp4 <= 0;
            calc_out_1_tmp5 <= 0;
            calc_out_1_tmp6 <= 0;
            calc_out_1_tmp7 <= 0;
            calc_out_1_tmp8 <= 0;
            calc_out_1_tmp9 <= 0;
            calc_out_1_tmp10 <= 0;
            calc_out_1_tmp11 <= 0;
            calc_out_1_tmp12 <= 0;
        end
        else
        begin
            calc_out_1_tmp0 <= exp_data[0]*weight_1[0] + exp_data[1]*weight_1[1];
            calc_out_1_tmp1 <= exp_data[2]*weight_1[2] + exp_data[3]*weight_1[3];
            calc_out_1_tmp2 <= exp_data[4]*weight_1[4] + exp_data[5]*weight_1[5];
            calc_out_1_tmp3 <= exp_data[6]*weight_1[6] + exp_data[7]*weight_1[7];
            calc_out_1_tmp4 <= exp_data[8]*weight_1[8] + exp_data[9]*weight_1[9];
            calc_out_1_tmp5 <= exp_data[10]*weight_1[10] + exp_data[11]*weight_1[11];
            calc_out_1_tmp6 <= exp_data[12]*weight_1[12] + exp_data[13]*weight_1[13];
            calc_out_1_tmp7 <= exp_data[14]*weight_1[14] + exp_data[15]*weight_1[15];
            calc_out_1_tmp8 <= exp_data[16]*weight_1[16] + exp_data[17]*weight_1[17];
            calc_out_1_tmp9 <= exp_data[18]*weight_1[18] + exp_data[19]*weight_1[19];
            calc_out_1_tmp10 <= exp_data[20]*weight_1[20] + exp_data[21]*weight_1[21];
            calc_out_1_tmp11 <= exp_data[22]*weight_1[22] + exp_data[23]*weight_1[23];
            calc_out_1_tmp12 <= exp_data[24]*weight_1[24];
        end
    end
    always @(posedge clk) // Stage 2
    begin
        if (~rst_n)
        begin
            calc_out_1_tmp13 <= 0;
            calc_out_1_tmp14 <= 0;
            calc_out_1_tmp15 <= 0;
            calc_out_1_tmp16 <= 0;
            calc_out_1_tmp17 <= 0;
            calc_out_1_tmp18 <= 0;
        end
        else
        begin
            calc_out_1_tmp13 <= calc_out_1_tmp0 + calc_out_1_tmp1;
            calc_out_1_tmp14 <= calc_out_1_tmp2 + calc_out_1_tmp3;
            calc_out_1_tmp15 <= calc_out_1_tmp4 + calc_out_1_tmp5;
            calc_out_1_tmp16 <= calc_out_1_tmp6 + calc_out_1_tmp7;
            calc_out_1_tmp17 <= calc_out_1_tmp8 + calc_out_1_tmp9;
            calc_out_1_tmp18 <= calc_out_1_tmp10 + calc_out_1_tmp11 + calc_out_1_tmp12;
        end
    end
    always @(posedge clk) // Stage 3
    begin
        if (~rst_n)
        begin
            calc_out_1_tmp19 <= 0;
            calc_out_1_tmp20 <= 0;
            calc_out_1_tmp21 <= 0;
        end
        else
        begin
            calc_out_1_tmp19 <= calc_out_1_tmp13 + calc_out_1_tmp14;
            calc_out_1_tmp20 <= calc_out_1_tmp15 + calc_out_1_tmp16;
            calc_out_1_tmp21 <= calc_out_1_tmp17 + calc_out_1_tmp18;
        end
    end
    always @(posedge clk) // Stage 4
    begin
        if (~rst_n)
        begin
            calc_out_1_tmp22 <= 0;
        end
        else
        begin
            calc_out_1_tmp22 <= calc_out_1_tmp19 + calc_out_1_tmp20 + calc_out_1_tmp21;
        end
    end
    assign calc_out_1 = calc_out_1_tmp22;

    // Multi stage multiplication and addition for out_2
    always @(posedge clk) // Stage 1
    begin
        if (~rst_n)
        begin
            calc_out_2_tmp0 <= 0;
            calc_out_2_tmp1 <= 0;
            calc_out_2_tmp2 <= 0;
            calc_out_2_tmp3 <= 0;
            calc_out_2_tmp4 <= 0;
            calc_out_2_tmp5 <= 0;
            calc_out_2_tmp6 <= 0;
            calc_out_2_tmp7 <= 0;
            calc_out_2_tmp8 <= 0;
            calc_out_2_tmp9 <= 0;
            calc_out_2_tmp10 <= 0;
            calc_out_2_tmp11 <= 0;
            calc_out_2_tmp12 <= 0;
        end
        else
        begin
            calc_out_2_tmp0 <= exp_data[0]*weight_2[0] + exp_data[1]*weight_2[1];
            calc_out_2_tmp1 <= exp_data[2]*weight_2[2] + exp_data[3]*weight_2[3];
            calc_out_2_tmp2 <= exp_data[4]*weight_2[4] + exp_data[5]*weight_2[5];
            calc_out_2_tmp3 <= exp_data[6]*weight_2[6] + exp_data[7]*weight_2[7];
            calc_out_2_tmp4 <= exp_data[8]*weight_2[8] + exp_data[9]*weight_2[9];
            calc_out_2_tmp5 <= exp_data[10]*weight_2[10] + exp_data[11]*weight_2[11];
            calc_out_2_tmp6 <= exp_data[12]*weight_2[12] + exp_data[13]*weight_2[13];
            calc_out_2_tmp7 <= exp_data[14]*weight_2[14] + exp_data[15]*weight_2[15];
            calc_out_2_tmp8 <= exp_data[16]*weight_2[16] + exp_data[17]*weight_2[17];
            calc_out_2_tmp9 <= exp_data[18]*weight_2[18] + exp_data[19]*weight_2[19];
            calc_out_2_tmp10 <= exp_data[20]*weight_2[20] + exp_data[21]*weight_2[21];
            calc_out_2_tmp11 <= exp_data[22]*weight_2[22] + exp_data[23]*weight_2[23];
            calc_out_2_tmp12 <= exp_data[24]*weight_2[24];
        end
    end
    always @(posedge clk) // Stage 2
    begin
        if (~rst_n)
        begin
            calc_out_2_tmp13 <= 0;
            calc_out_2_tmp14 <= 0;
            calc_out_2_tmp15 <= 0;
            calc_out_2_tmp16 <= 0;
            calc_out_2_tmp17 <= 0;
            calc_out_2_tmp18 <= 0;
        end
        else
        begin
            calc_out_2_tmp13 <= calc_out_2_tmp0 + calc_out_2_tmp1;
            calc_out_2_tmp14 <= calc_out_2_tmp2 + calc_out_2_tmp3;
            calc_out_2_tmp15 <= calc_out_2_tmp4 + calc_out_2_tmp5;
            calc_out_2_tmp16 <= calc_out_2_tmp6 + calc_out_2_tmp7;
            calc_out_2_tmp17 <= calc_out_2_tmp8 + calc_out_2_tmp9;
            calc_out_2_tmp18 <= calc_out_2_tmp10 + calc_out_2_tmp11 + calc_out_2_tmp12;
        end
    end
    always @(posedge clk) // Stage 3
    begin
        if (~rst_n)
        begin
            calc_out_2_tmp19 <= 0;
            calc_out_2_tmp20 <= 0;
            calc_out_2_tmp21 <= 0;
        end
        else
        begin
            calc_out_2_tmp19 <= calc_out_2_tmp13 + calc_out_2_tmp14;
            calc_out_2_tmp20 <= calc_out_2_tmp15 + calc_out_2_tmp16;
            calc_out_2_tmp21 <= calc_out_2_tmp17 + calc_out_2_tmp18;
        end
    end
    always @(posedge clk) // Stage 4
    begin
        if (~rst_n)
        begin
            calc_out_2_tmp22 <= 0;
        end
        else
        begin
            calc_out_2_tmp22 <= calc_out_2_tmp19 + calc_out_2_tmp20 + calc_out_2_tmp21;
        end
    end
    assign calc_out_2 = calc_out_2_tmp22;

    // Multi stage multiplication and addition for out_3
    always @(posedge clk) // Stage 1
    begin
        if (~rst_n)
        begin
            calc_out_3_tmp0 <= 0;
            calc_out_3_tmp1 <= 0;
            calc_out_3_tmp2 <= 0;
            calc_out_3_tmp3 <= 0;
            calc_out_3_tmp4 <= 0;
            calc_out_3_tmp5 <= 0;
            calc_out_3_tmp6 <= 0;
            calc_out_3_tmp7 <= 0;
            calc_out_3_tmp8 <= 0;
            calc_out_3_tmp9 <= 0;
            calc_out_3_tmp10 <= 0;
            calc_out_3_tmp11 <= 0;
            calc_out_3_tmp12 <= 0;
        end
        else
        begin
            calc_out_3_tmp0 <= exp_data[0]*weight_3[0] + exp_data[1]*weight_3[1];
            calc_out_3_tmp1 <= exp_data[2]*weight_3[2] + exp_data[3]*weight_3[3];
            calc_out_3_tmp2 <= exp_data[4]*weight_3[4] + exp_data[5]*weight_3[5];
            calc_out_3_tmp3 <= exp_data[6]*weight_3[6] + exp_data[7]*weight_3[7];
            calc_out_3_tmp4 <= exp_data[8]*weight_3[8] + exp_data[9]*weight_3[9];
            calc_out_3_tmp5 <= exp_data[10]*weight_3[10] + exp_data[11]*weight_3[11];
            calc_out_3_tmp6 <= exp_data[12]*weight_3[12] + exp_data[13]*weight_3[13];
            calc_out_3_tmp7 <= exp_data[14]*weight_3[14] + exp_data[15]*weight_3[15];
            calc_out_3_tmp8 <= exp_data[16]*weight_3[16] + exp_data[17]*weight_3[17];
            calc_out_3_tmp9 <= exp_data[18]*weight_3[18] + exp_data[19]*weight_3[19];
            calc_out_3_tmp10 <= exp_data[20]*weight_3[20] + exp_data[21]*weight_3[21];
            calc_out_3_tmp11 <= exp_data[22]*weight_3[22] + exp_data[23]*weight_3[23];
            calc_out_3_tmp12 <= exp_data[24]*weight_3[24];
        end
    end
    always @(posedge clk) // Stage 2
    begin
        if (~rst_n)
        begin
            calc_out_3_tmp13 <= 0;
            calc_out_3_tmp14 <= 0;
            calc_out_3_tmp15 <= 0;
            calc_out_3_tmp16 <= 0;
            calc_out_3_tmp17 <= 0;
            calc_out_3_tmp18 <= 0;
        end
        else
        begin
            calc_out_3_tmp13 <= calc_out_3_tmp0 + calc_out_3_tmp1;
            calc_out_3_tmp14 <= calc_out_3_tmp2 + calc_out_3_tmp3;
            calc_out_3_tmp15 <= calc_out_3_tmp4 + calc_out_3_tmp5;
            calc_out_3_tmp16 <= calc_out_3_tmp6 + calc_out_3_tmp7;
            calc_out_3_tmp17 <= calc_out_3_tmp8 + calc_out_3_tmp9;
            calc_out_3_tmp18 <= calc_out_3_tmp10 + calc_out_3_tmp11 + calc_out_3_tmp12;
        end
    end
    always @(posedge clk) // Stage 3
    begin
        if (~rst_n)
        begin
            calc_out_3_tmp19 <= 0;
            calc_out_3_tmp20 <= 0;
            calc_out_3_tmp21 <= 0;
        end
        else
        begin
            calc_out_3_tmp19 <= calc_out_3_tmp13 + calc_out_3_tmp14;
            calc_out_3_tmp20 <= calc_out_3_tmp15 + calc_out_3_tmp16;
            calc_out_3_tmp21 <= calc_out_3_tmp17 + calc_out_3_tmp18;
        end
    end
    always @(posedge clk) // Stage 4
    begin
        if (~rst_n)
        begin
            calc_out_3_tmp22 <= 0;
        end
        else 
        begin
            calc_out_3_tmp22 <= calc_out_3_tmp19 + calc_out_3_tmp20 + calc_out_3_tmp21;
        end
    end
    assign calc_out_3 = calc_out_3_tmp22;

    assign conv_out_1 = calc_out_1[19:8] + exp_bias[0];
    assign conv_out_2 = calc_out_2[19:8] + exp_bias[1];
    assign conv_out_3 = calc_out_3[19:8] + exp_bias[2];

    always @(posedge clk)
    begin
        if (~rst_n)
        begin
            valid_out_buf_tmp0 <= 0;
            valid_out_buf_tmp1 <= 0;
            valid_out_buf_tmp2 <= 0;
            valid_out_buf_tmp3 <= 0;
        end
        else
        begin
            valid_out_buf_tmp0 <= valid_out_buf;
            valid_out_buf_tmp1 <= valid_out_buf_tmp0;
            valid_out_buf_tmp2 <= valid_out_buf_tmp1;
            valid_out_buf_tmp3 <= valid_out_buf_tmp2;
        end
    end

    assign valid_out_calc = valid_out_buf_tmp3;

endmodule
