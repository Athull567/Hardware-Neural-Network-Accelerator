/*------------------------------------------------------------------------
 *
 *  Copyright (c) 2021 by Bo Young Kang, All rights reserved.
 *
 *  File name  : conv2_buf.v
 *  Written by : Kang, Bo Young, weenslab
 *  Written on : Oct 13, 2021
 *  Version    : 22
 *  Design     : 2nd Convolution Layer for CNN MNIST dataset
 *               Input Buffer
 *
 *------------------------------------------------------------------------*/

/*-------------------------------------------------------------------
 *  Module: conv2_buf
 *------------------------------------------------------------------*/

module conv2_buf
    #(
        parameter WIDTH = 12,
                  HEIGHT = 12,
                  DATA_BITS = 12
    )
    (
        input  wire                 clk,
        input  wire                 rst_n,
        input  wire                 valid_in,
        input  wire [DATA_BITS-1:0] data_in,
        output reg  [DATA_BITS-1:0] data_out_0, data_out_1, data_out_2, data_out_3, data_out_4,
                                    data_out_5, data_out_6, data_out_7, data_out_8, data_out_9,
                                    data_out_10, data_out_11, data_out_12, data_out_13, data_out_14,
                                    data_out_15, data_out_16, data_out_17, data_out_18, data_out_19,
                                    data_out_20, data_out_21, data_out_22, data_out_23, data_out_24,
        output reg                  valid_out_buf
    );

    localparam FILTER_SIZE = 5;
    
    reg [DATA_BITS-1:0] buffer [0:WIDTH*FILTER_SIZE-1];
    reg [DATA_BITS-1:0] buf_idx;
    reg [4:0]           w_idx, h_idx;
    reg [2:0]           buf_flag; // 0 ~ 4
    reg                 state;

    integer i;

    always @(posedge clk)
    begin
        if (~rst_n)
        begin
            for (i=0; i <= WIDTH*FILTER_SIZE-1; i=i+1)
            begin
                buffer[i] <= 0;
            end
            buf_idx <= 0;
            w_idx <= 0;
            h_idx <= 0;
            buf_flag <= 0;
            state <= 0;
            valid_out_buf <= 0;
            data_out_0 <= 0;
            data_out_1 <= 0;
            data_out_2 <= 0;
            data_out_3 <= 0;
            data_out_4 <= 0;
            data_out_5 <= 0;
            data_out_6 <= 0;
            data_out_7 <= 0;
            data_out_8 <= 0;
            data_out_9 <= 0;
            data_out_10 <= 0;
            data_out_11 <= 0;
            data_out_12 <= 0;
            data_out_13 <= 0;
            data_out_14 <= 0;
            data_out_15 <= 0;
            data_out_16 <= 0;
            data_out_17 <= 0;
            data_out_18 <= 0;
            data_out_19 <= 0;
            data_out_20 <= 0;
            data_out_21 <= 0;
            data_out_22 <= 0;
            data_out_23 <= 0;
            data_out_24 <= 0;
        end
        else
        begin
            if (valid_in)
            begin
                buf_idx <= buf_idx + 1'b1;
                if (buf_idx == WIDTH*FILTER_SIZE-1)
                begin // buffer size = 140 = 28(w) * 5(h)
                    buf_idx <= 0;
                end

                buffer[buf_idx] <= data_in; // data input

                // Wait until first 140 input data filled in buffer
                if (!state)
                begin
                    if (buf_idx == WIDTH*FILTER_SIZE-1)
                    begin
                        state <= 1;
                    end
                end
                else
                begin // valid state
                    w_idx <= w_idx + 1'b1; // move right

                    if (w_idx == WIDTH-FILTER_SIZE+1)
                    begin
                        valid_out_buf <= 1'b0;  // unvalid area
                    end
                    else if (w_idx == WIDTH-1)
                    begin
                        buf_flag <= buf_flag + 1;
                        if (buf_flag == FILTER_SIZE-1)
                        begin
                            buf_flag <= 0;
                        end
                        w_idx <= 0;

                        if (h_idx == HEIGHT-FILTER_SIZE)
                        begin // done 1 input read -> 28 * 28
                            h_idx <= 0;
                            state <= 0;
                        end
                        h_idx <= h_idx + 1;
                    end
                    else if (w_idx == 0)
                    begin
                        valid_out_buf <= 1'b1; // start valid area
                    end

                    // Buffer Selection -> 5 * 5
                    if (buf_flag == 3'd0)
                    begin
                        data_out_0 <= buffer[w_idx];
                        data_out_1 <= buffer[w_idx + 1];
                        data_out_2 <= buffer[w_idx + 2];
                        data_out_3 <= buffer[w_idx + 3];
                        data_out_4 <= buffer[w_idx + 4];

                        data_out_5 <= buffer[w_idx + WIDTH];
                        data_out_6 <= buffer[w_idx + 1 + WIDTH];
                        data_out_7 <= buffer[w_idx + 2 + WIDTH];
                        data_out_8 <= buffer[w_idx + 3 + WIDTH];
                        data_out_9 <= buffer[w_idx + 4 + WIDTH];

                        data_out_10 <= buffer[w_idx + WIDTH * 2];
                        data_out_11 <= buffer[w_idx + 1 + WIDTH * 2];
                        data_out_12 <= buffer[w_idx + 2 + WIDTH * 2];
                        data_out_13 <= buffer[w_idx + 3 + WIDTH * 2];
                        data_out_14 <= buffer[w_idx + 4 + WIDTH * 2];

                        data_out_15 <= buffer[w_idx + WIDTH * 3];
                        data_out_16 <= buffer[w_idx + 1 + WIDTH * 3];
                        data_out_17 <= buffer[w_idx + 2 + WIDTH * 3];
                        data_out_18 <= buffer[w_idx + 3 + WIDTH * 3];
                        data_out_19 <= buffer[w_idx + 4 + WIDTH * 3];

                        data_out_20 <= buffer[w_idx + WIDTH * 4];
                        data_out_21 <= buffer[w_idx + 1 + WIDTH * 4];
                        data_out_22 <= buffer[w_idx + 2 + WIDTH * 4];
                        data_out_23 <= buffer[w_idx + 3 + WIDTH * 4];
                        data_out_24 <= buffer[w_idx + 4 + WIDTH * 4];
                    end
                    else if (buf_flag == 3'd1)
                    begin
                        data_out_0 <= buffer[w_idx + WIDTH];
                        data_out_1 <= buffer[w_idx + 1 + WIDTH];
                        data_out_2 <= buffer[w_idx + 2 + WIDTH];
                        data_out_3 <= buffer[w_idx + 3 + WIDTH];
                        data_out_4 <= buffer[w_idx + 4 + WIDTH];

                        data_out_5 <= buffer[w_idx + WIDTH * 2];
                        data_out_6 <= buffer[w_idx + 1 + WIDTH * 2];
                        data_out_7 <= buffer[w_idx + 2 + WIDTH * 2];
                        data_out_8 <= buffer[w_idx + 3 + WIDTH * 2];
                        data_out_9 <= buffer[w_idx + 4 + WIDTH * 2];

                        data_out_10 <= buffer[w_idx + WIDTH * 3];
                        data_out_11 <= buffer[w_idx + 1 + WIDTH * 3];
                        data_out_12 <= buffer[w_idx + 2 + WIDTH * 3];
                        data_out_13 <= buffer[w_idx + 3 + WIDTH * 3];
                        data_out_14 <= buffer[w_idx + 4 + WIDTH * 3];

                        data_out_15 <= buffer[w_idx + WIDTH * 4];
                        data_out_16 <= buffer[w_idx + 1 + WIDTH * 4];
                        data_out_17 <= buffer[w_idx + 2 + WIDTH * 4];
                        data_out_18 <= buffer[w_idx + 3 + WIDTH * 4];
                        data_out_19 <= buffer[w_idx + 4 + WIDTH * 4];

                        data_out_20 <= buffer[w_idx];
                        data_out_21 <= buffer[w_idx + 1];
                        data_out_22 <= buffer[w_idx + 2];
                        data_out_23 <= buffer[w_idx + 3];
                        data_out_24 <= buffer[w_idx + 4];
                    end
                    else if (buf_flag == 3'd2)
                    begin
                        data_out_0 <= buffer[w_idx + WIDTH * 2];
                        data_out_1 <= buffer[w_idx + 1 + WIDTH * 2];
                        data_out_2 <= buffer[w_idx + 2 + WIDTH * 2];
                        data_out_3 <= buffer[w_idx + 3 + WIDTH * 2];
                        data_out_4 <= buffer[w_idx + 4 + WIDTH * 2];

                        data_out_5 <= buffer[w_idx + WIDTH * 3];
                        data_out_6 <= buffer[w_idx + 1 + WIDTH * 3];
                        data_out_7 <= buffer[w_idx + 2 + WIDTH * 3];
                        data_out_8 <= buffer[w_idx + 3 + WIDTH * 3];
                        data_out_9 <= buffer[w_idx + 4 + WIDTH * 3];

                        data_out_10 <= buffer[w_idx + WIDTH * 4];
                        data_out_11 <= buffer[w_idx + 1 + WIDTH * 4];
                        data_out_12 <= buffer[w_idx + 2 + WIDTH * 4];
                        data_out_13 <= buffer[w_idx + 3 + WIDTH * 4];
                        data_out_14 <= buffer[w_idx + 4 + WIDTH * 4];

                        data_out_15 <= buffer[w_idx];
                        data_out_16 <= buffer[w_idx + 1];
                        data_out_17 <= buffer[w_idx + 2];
                        data_out_18 <= buffer[w_idx + 3];
                        data_out_19 <= buffer[w_idx + 4];

                        data_out_20 <= buffer[w_idx + WIDTH];
                        data_out_21 <= buffer[w_idx + 1 + WIDTH];
                        data_out_22 <= buffer[w_idx + 2 + WIDTH];
                        data_out_23 <= buffer[w_idx + 3 + WIDTH];
                        data_out_24 <= buffer[w_idx + 4 + WIDTH];
                    end
                    else if (buf_flag == 3'd3)
                    begin
                        data_out_0 <= buffer[w_idx + WIDTH * 3];
                        data_out_1 <= buffer[w_idx + 1 + WIDTH * 3];
                        data_out_2 <= buffer[w_idx + 2 + WIDTH * 3];
                        data_out_3 <= buffer[w_idx + 3 + WIDTH * 3];
                        data_out_4 <= buffer[w_idx + 4 + WIDTH * 3];

                        data_out_5 <= buffer[w_idx + WIDTH * 4];
                        data_out_6 <= buffer[w_idx + 1 + WIDTH * 4];
                        data_out_7 <= buffer[w_idx + 2 + WIDTH * 4];
                        data_out_8 <= buffer[w_idx + 3 + WIDTH * 4];
                        data_out_9 <= buffer[w_idx + 4 + WIDTH * 4];

                        data_out_10 <= buffer[w_idx];
                        data_out_11 <= buffer[w_idx + 1];
                        data_out_12 <= buffer[w_idx + 2];
                        data_out_13 <= buffer[w_idx + 3];
                        data_out_14 <= buffer[w_idx + 4];

                        data_out_15 <= buffer[w_idx + WIDTH];
                        data_out_16 <= buffer[w_idx + 1 + WIDTH];
                        data_out_17 <= buffer[w_idx + 2 + WIDTH];
                        data_out_18 <= buffer[w_idx + 3 + WIDTH];
                        data_out_19 <= buffer[w_idx + 4 + WIDTH];

                        data_out_20 <= buffer[w_idx + WIDTH * 2];
                        data_out_21 <= buffer[w_idx + 1 + WIDTH * 2];
                        data_out_22 <= buffer[w_idx + 2 + WIDTH * 2];
                        data_out_23 <= buffer[w_idx + 3 + WIDTH * 2];
                        data_out_24 <= buffer[w_idx + 4 + WIDTH * 2];
                    end
                    else if (buf_flag == 3'd4)
                    begin
                        data_out_0 <= buffer[w_idx + WIDTH * 4];
                        data_out_1 <= buffer[w_idx + 1 + WIDTH * 4];
                        data_out_2 <= buffer[w_idx + 2 + WIDTH * 4];
                        data_out_3 <= buffer[w_idx + 3 + WIDTH * 4];
                        data_out_4 <= buffer[w_idx + 4 + WIDTH * 4];

                        data_out_5 <= buffer[w_idx];
                        data_out_6 <= buffer[w_idx + 1];
                        data_out_7 <= buffer[w_idx + 2];
                        data_out_8 <= buffer[w_idx + 3];
                        data_out_9 <= buffer[w_idx + 4];

                        data_out_10 <= buffer[w_idx + WIDTH];
                        data_out_11 <= buffer[w_idx + 1 + WIDTH];
                        data_out_12 <= buffer[w_idx + 2 + WIDTH];
                        data_out_13 <= buffer[w_idx + 3 + WIDTH];
                        data_out_14 <= buffer[w_idx + 4 + WIDTH];

                        data_out_15 <= buffer[w_idx + WIDTH * 2];
                        data_out_16 <= buffer[w_idx + 1 + WIDTH * 2];
                        data_out_17 <= buffer[w_idx + 2 + WIDTH * 2];
                        data_out_18 <= buffer[w_idx + 3 + WIDTH * 2];
                        data_out_19 <= buffer[w_idx + 4 + WIDTH * 2];

                        data_out_20 <= buffer[w_idx + WIDTH * 3];
                        data_out_21 <= buffer[w_idx + 1 + WIDTH * 3];
                        data_out_22 <= buffer[w_idx + 2 + WIDTH * 3];
                        data_out_23 <= buffer[w_idx + 3 + WIDTH * 3];
                        data_out_24 <= buffer[w_idx + 4 + WIDTH * 3];
                    end
                end
            end
        end
     end

endmodule
