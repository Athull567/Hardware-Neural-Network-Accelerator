/*------------------------------------------------------------------------
 *
 *  Copyright (c) 2021 by Bo Young Kang, All rights reserved.
 *
 *  File name  : fully_connected.v
 *  Written by : Kang, Bo Young, weenslab
 *  Written on : Oct 13, 2021
 *  Version    : 22
 *  Design     : Fully Connected Layer for CNN
 *
 *------------------------------------------------------------------------*/

/*-------------------------------------------------------------------
 *  Module: fully_connected
 *------------------------------------------------------------------*/

module fully_connected
    #(
        parameter INPUT_NUM = 48,
                  OUTPUT_NUM = 10,
                  DATA_BITS = 8
    )
    (
        input  wire               clk,
        input  wire               rst_n,
        input  wire               valid_in,
        input  wire signed [11:0] data_in_1, data_in_2, data_in_3,
        output wire [11:0]        data_out,
        output wire               valid_out_fc
    );

    localparam INPUT_WIDTH = 16;
    localparam INPUT_NUM_DATA_BITS = 5;

    reg                   state;
    reg [INPUT_WIDTH-1:0] buf_idx;
    reg [3:0]             out_idx;

    reg signed [13:0]          buffer [0:INPUT_NUM-1];
    reg signed [DATA_BITS-1:0] weight [0:INPUT_NUM*OUTPUT_NUM-1];
    reg signed [DATA_BITS-1:0] bias   [0:OUTPUT_NUM-1];

    reg signed [19:0] calc_out_tmp0, calc_out_tmp1, calc_out_tmp2, calc_out_tmp3,
                      calc_out_tmp4, calc_out_tmp5, calc_out_tmp6, calc_out_tmp7,
                      calc_out_tmp8, calc_out_tmp9, calc_out_tmp10, calc_out_tmp11,
                      calc_out_tmp12, calc_out_tmp13, calc_out_tmp14, calc_out_tmp15,
                      calc_out_tmp16, calc_out_tmp17, calc_out_tmp18, calc_out_tmp19,
                      calc_out_tmp20, calc_out_tmp21, calc_out_tmp22, calc_out_tmp23,
                      calc_out_tmp24, calc_out_tmp25, calc_out_tmp26, calc_out_tmp27,
                      calc_out_tmp28, calc_out_tmp29, calc_out_tmp30, calc_out_tmp31,
                      calc_out_tmp32, calc_out_tmp33, calc_out_tmp34, calc_out_tmp35,
                      calc_out_tmp36, calc_out_tmp37, calc_out_tmp38, calc_out_tmp39,
                      calc_out_tmp40, calc_out_tmp41, calc_out_tmp42, calc_out_tmp43,
                      calc_out_tmp44, calc_out_tmp45, calc_out_tmp46;

    wire signed [19:0] calc_out;
    wire signed [13:0] data1, data2, data3;

    reg valid_out_fc_tmp0, valid_out_fc_tmp1, valid_out_fc_tmp2, valid_out_fc_tmp3,
        valid_out_fc_tmp4, valid_out_fc_tmp5;

    initial
    begin
        $readmemh("fc_weight.mem", weight);
        $readmemh("fc_bias.mem", bias);
    end

    assign data1 = (data_in_1[11] == 1) ? {2'b11, data_in_1} : {2'b00, data_in_1};
    assign data2 = (data_in_2[11] == 1) ? {2'b11, data_in_2} : {2'b00, data_in_2};
    assign data3 = (data_in_3[11] == 1) ? {2'b11, data_in_3} : {2'b00, data_in_3};

    integer i;

    always @(posedge clk)
    begin
        if (~rst_n)
        begin
            for (i=0; i <= INPUT_NUM-1; i=i+1)
            begin
                buffer[i] <= 0;
            end
            valid_out_fc_tmp0 <= 0;
            buf_idx <= 0;
            out_idx <= 0;
            state <= 0;
        end
        else
        begin
            if (valid_out_fc_tmp0 == 1)
            begin
                valid_out_fc_tmp0 <= 0;
            end

            if (valid_in == 1)
            begin
                // Wait until 48 input data filled in buffer
                if (!state)
                begin
                    buffer[buf_idx] <= data1;
                    buffer[INPUT_WIDTH + buf_idx] <= data2;
                    buffer[INPUT_WIDTH * 2 + buf_idx] <= data3;
                    buf_idx <= buf_idx + 1'b1;
                    if (buf_idx == INPUT_WIDTH-1)
                    begin
                        buf_idx <= 0;
                        state <= 1;
                        valid_out_fc_tmp0 <= 1;
                    end
                end
                else
                begin // valid state
                    out_idx <= out_idx + 1'b1;
                    if (out_idx == OUTPUT_NUM-1)
                    begin
                        out_idx <= 0;
                    end
                    valid_out_fc_tmp0 <= 1;
                end
            end
        end
    end

    // Multi stage multiplication and addition
    always @(posedge clk)
    begin
        if (~rst_n)
        begin
            calc_out_tmp0 <= 0;
            calc_out_tmp1 <= 0;
            calc_out_tmp2 <= 0;
            calc_out_tmp3 <= 0;
            calc_out_tmp4 <= 0;
            calc_out_tmp5 <= 0;
            calc_out_tmp6 <= 0;
            calc_out_tmp7 <= 0;
            calc_out_tmp8 <= 0;
            calc_out_tmp9 <= 0;
            calc_out_tmp10 <= 0;
            calc_out_tmp11 <= 0;
            calc_out_tmp12 <= 0;
            calc_out_tmp13 <= 0;
            calc_out_tmp14 <= 0;
            calc_out_tmp15 <= 0;
            calc_out_tmp16 <= 0;
            calc_out_tmp17 <= 0;
            calc_out_tmp18 <= 0;
            calc_out_tmp19 <= 0;
            calc_out_tmp20 <= 0;
            calc_out_tmp21 <= 0;
            calc_out_tmp22 <= 0;
            calc_out_tmp23 <= 0;
            calc_out_tmp24 <= 0;
        end
        else
        begin
            calc_out_tmp0 <= weight[out_idx * INPUT_NUM] * buffer[0] + weight[out_idx * INPUT_NUM + 1] * buffer[1];
            calc_out_tmp1 <= weight[out_idx * INPUT_NUM + 2] * buffer[2] + weight[out_idx * INPUT_NUM + 3] * buffer[3];
            calc_out_tmp2 <= weight[out_idx * INPUT_NUM + 4] * buffer[4] + weight[out_idx * INPUT_NUM + 5] * buffer[5];
            calc_out_tmp3 <= weight[out_idx * INPUT_NUM + 6] * buffer[6] + weight[out_idx * INPUT_NUM + 7] * buffer[7];
            calc_out_tmp4 <= weight[out_idx * INPUT_NUM + 8] * buffer[8] + weight[out_idx * INPUT_NUM + 9] * buffer[9];
            calc_out_tmp5 <= weight[out_idx * INPUT_NUM + 10] * buffer[10] + weight[out_idx * INPUT_NUM + 11] * buffer[11];
            calc_out_tmp6 <= weight[out_idx * INPUT_NUM + 12] * buffer[12] + weight[out_idx * INPUT_NUM + 13] * buffer[13];
            calc_out_tmp7 <= weight[out_idx * INPUT_NUM + 14] * buffer[14] + weight[out_idx * INPUT_NUM + 15] * buffer[15];
            calc_out_tmp8 <= weight[out_idx * INPUT_NUM + 16] * buffer[16] + weight[out_idx * INPUT_NUM + 17] * buffer[17];
            calc_out_tmp9 <= weight[out_idx * INPUT_NUM + 18] * buffer[18] + weight[out_idx * INPUT_NUM + 19] * buffer[19];
            calc_out_tmp10 <= weight[out_idx * INPUT_NUM + 20] * buffer[20] + weight[out_idx * INPUT_NUM + 21] * buffer[21];
            calc_out_tmp11 <= weight[out_idx * INPUT_NUM + 22] * buffer[22] + weight[out_idx * INPUT_NUM + 23] * buffer[23];
            calc_out_tmp12 <= weight[out_idx * INPUT_NUM + 24] * buffer[24] + weight[out_idx * INPUT_NUM + 25] * buffer[25];
            calc_out_tmp13 <= weight[out_idx * INPUT_NUM + 26] * buffer[26] + weight[out_idx * INPUT_NUM + 27] * buffer[27];
            calc_out_tmp14 <= weight[out_idx * INPUT_NUM + 28] * buffer[28] + weight[out_idx * INPUT_NUM + 29] * buffer[29];
            calc_out_tmp15 <= weight[out_idx * INPUT_NUM + 30] * buffer[30] + weight[out_idx * INPUT_NUM + 31] * buffer[31];
            calc_out_tmp16 <= weight[out_idx * INPUT_NUM + 32] * buffer[32] + weight[out_idx * INPUT_NUM + 33] * buffer[33];
            calc_out_tmp17 <= weight[out_idx * INPUT_NUM + 34] * buffer[34] + weight[out_idx * INPUT_NUM + 35] * buffer[35];
            calc_out_tmp18 <= weight[out_idx * INPUT_NUM + 36] * buffer[36] + weight[out_idx * INPUT_NUM + 37] * buffer[37];
            calc_out_tmp19 <= weight[out_idx * INPUT_NUM + 38] * buffer[38] + weight[out_idx * INPUT_NUM + 39] * buffer[39];
            calc_out_tmp20 <= weight[out_idx * INPUT_NUM + 40] * buffer[40] + weight[out_idx * INPUT_NUM + 41] * buffer[41];
            calc_out_tmp21 <= weight[out_idx * INPUT_NUM + 42] * buffer[42] + weight[out_idx * INPUT_NUM + 43] * buffer[43];
            calc_out_tmp22 <= weight[out_idx * INPUT_NUM + 44] * buffer[44] + weight[out_idx * INPUT_NUM + 45] * buffer[45];
            calc_out_tmp23 <= weight[out_idx * INPUT_NUM + 46] * buffer[46] + weight[out_idx * INPUT_NUM + 47] * buffer[47];
            calc_out_tmp24 <= bias[out_idx];
        end
    end
    always @(posedge clk)
    begin
        if (~rst_n)
        begin
            calc_out_tmp25 <= 0;
            calc_out_tmp26 <= 0;
            calc_out_tmp27 <= 0;
            calc_out_tmp28 <= 0;
            calc_out_tmp29 <= 0;
            calc_out_tmp30 <= 0;
            calc_out_tmp31 <= 0;
            calc_out_tmp32 <= 0;
            calc_out_tmp33 <= 0;
            calc_out_tmp34 <= 0;
            calc_out_tmp35 <= 0;
            calc_out_tmp36 <= 0;
        end
        else
        begin
            calc_out_tmp25 <= calc_out_tmp0 + calc_out_tmp1;
            calc_out_tmp26 <= calc_out_tmp2 + calc_out_tmp3;
            calc_out_tmp27 <= calc_out_tmp4 + calc_out_tmp5;
            calc_out_tmp28 <= calc_out_tmp6 + calc_out_tmp7;
            calc_out_tmp29 <= calc_out_tmp8 + calc_out_tmp9;
            calc_out_tmp30 <= calc_out_tmp10 + calc_out_tmp11;
            calc_out_tmp31 <= calc_out_tmp12 + calc_out_tmp13;
            calc_out_tmp32 <= calc_out_tmp14 + calc_out_tmp15;
            calc_out_tmp33 <= calc_out_tmp16 + calc_out_tmp17;
            calc_out_tmp34 <= calc_out_tmp18 + calc_out_tmp19;
            calc_out_tmp35 <= calc_out_tmp20 + calc_out_tmp21;
            calc_out_tmp36 <= calc_out_tmp22 + calc_out_tmp23 + calc_out_tmp24;
        end
    end
    always @(posedge clk)
    begin
        if (~rst_n)
        begin
            calc_out_tmp37 <= 0;
            calc_out_tmp38 <= 0;
            calc_out_tmp39 <= 0;
            calc_out_tmp40 <= 0;
            calc_out_tmp41 <= 0;
            calc_out_tmp42 <= 0;
        end
        else
        begin
            calc_out_tmp37 <= calc_out_tmp25 + calc_out_tmp26;
            calc_out_tmp38 <= calc_out_tmp27 + calc_out_tmp28;
            calc_out_tmp39 <= calc_out_tmp29 + calc_out_tmp30;
            calc_out_tmp40 <= calc_out_tmp31 + calc_out_tmp32;
            calc_out_tmp41 <= calc_out_tmp33 + calc_out_tmp34;
            calc_out_tmp42 <= calc_out_tmp35 + calc_out_tmp36;
        end
    end
    always @(posedge clk)
    begin
        if (~rst_n)
        begin
            calc_out_tmp43 <= 0;
            calc_out_tmp44 <= 0;
            calc_out_tmp45 <= 0;
        end
        else
        begin
            calc_out_tmp43 <= calc_out_tmp37 + calc_out_tmp38;
            calc_out_tmp44 <= calc_out_tmp39 + calc_out_tmp40;
            calc_out_tmp45 <= calc_out_tmp41 + calc_out_tmp42;
        end
    end
    always @(posedge clk)
    begin
        if (~rst_n)
        begin
            calc_out_tmp46 <= 0;
        end
        else
        begin
            calc_out_tmp46 <= calc_out_tmp43 + calc_out_tmp44 + calc_out_tmp45;
        end
    end

    assign calc_out = calc_out_tmp46;
    assign data_out = calc_out[18:7];

    always @(posedge clk)
    begin
        if (~rst_n)
        begin
            valid_out_fc_tmp1 <= 0;
            valid_out_fc_tmp2 <= 0;
            valid_out_fc_tmp3 <= 0;
            valid_out_fc_tmp4 <= 0;
            valid_out_fc_tmp5 <= 0;
        end
        else
        begin
            valid_out_fc_tmp1 <= valid_out_fc_tmp0;
            valid_out_fc_tmp2 <= valid_out_fc_tmp1;
            valid_out_fc_tmp3 <= valid_out_fc_tmp2;
            valid_out_fc_tmp4 <= valid_out_fc_tmp3;
            valid_out_fc_tmp5 <= valid_out_fc_tmp4;
        end
    end

    assign valid_out_fc = valid_out_fc_tmp5;

endmodule
