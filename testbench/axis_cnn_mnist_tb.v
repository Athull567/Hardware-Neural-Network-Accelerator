`timescale 1ns / 10ps

module axis_cnn_mnist_tb();
    localparam T = 10; // 100 MHz

    reg aclk;
    reg aresetn;
    wire s_axis_tready;
    reg [7:0] s_axis_tdata;
    reg s_axis_tvalid;
    reg m_axis_tready;
    wire [7:0] m_axis_tdata;
    wire m_axis_tvalid;
    wire m_axis_tlast;

    reg [7:0] s_axis_tdata_reg;
    reg s_axis_tvalid_reg;

    reg [7:0] pixels [0:783];
    integer i = 0;

    axis_cnn_mnist dut
    (
        .aclk(aclk),
        .aresetn(aresetn),
        .s_axis_tready(s_axis_tready),
        .s_axis_tdata(s_axis_tdata_reg),
        .s_axis_tvalid(s_axis_tvalid_reg),
        .m_axis_tready(m_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tlast(m_axis_tlast)
    );

    always
    begin
        aclk = 0;
        #(T/2);
        aclk = 1;
        #(T/2);
    end

    initial
    begin
        $readmemh("../../../../rtl/testvector/3_0.txt", pixels);

        s_axis_tdata = 0;
        s_axis_tvalid = 0;
        m_axis_tready = 1;

        aresetn = 1'b0;
        #(T*5);
        aresetn = 1'b1;
        #(T*5);
        
        #(T*20);
        s_axis_tvalid = 1;
        for (i = 0; i < 784; i=i+1)
        begin
            s_axis_tdata = pixels[i];
            #T;
        end
        s_axis_tvalid = 0;
        while (s_axis_tready == 1)
        begin
            #T;
        end
        while (s_axis_tready == 0)
        begin
            #T;
        end
        
        s_axis_tvalid = 1;
        for (i = 0; i < 784; i=i+1)
        begin
            s_axis_tdata = pixels[i];
            #T;
        end
        s_axis_tvalid = 0;
        while (s_axis_tready == 1)
        begin
            #T;
        end
        while (s_axis_tready == 0)
        begin
            #T;
        end
    end

    always @(posedge aclk)
    begin
        if (!aresetn)
        begin
            s_axis_tdata_reg <= 0;
            s_axis_tvalid_reg <= 0;
        end
        else
        begin
            s_axis_tdata_reg <= s_axis_tdata;
            s_axis_tvalid_reg <= s_axis_tvalid;
        end
    end

endmodule
