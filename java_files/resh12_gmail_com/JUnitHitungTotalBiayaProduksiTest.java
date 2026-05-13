package resh12_gmail_com;

import static org.junit.Assert.*;

import java.io.ByteArrayOutputStream;

import java.io.PrintStream;

import org.junit.After;

import org.junit.Before;

import org.junit.Test;



public class JUnitHitungTotalBiayaProduksiTest {

    

    private ByteArrayOutputStream outContent;

    private PrintStream originalOut;



    @Before

    public void setUpStreams() {

        outContent = new ByteArrayOutputStream();

        originalOut = System.out;

        System.setOut(new PrintStream(outContent));

    }



    @After

    public void restoreStreams() {

        System.setOut(originalOut);

    }



    @Test

    public void testMainMethodExecution() {

        // Test that main method runs without errors

        try {

            HitungTotalBiayaProduksi.main(null); 

            String output = outContent.toString().trim();

            assertFalse("Program harus menghasilkan output", output.isEmpty());

        } catch (Exception e) {

            fail("Program tidak boleh error: " + e.getMessage());

        }

    }



    @Test

    public void testActualOutput() {

        // Test the exact output the program produces

        HitungTotalBiayaProduksi.main(null);

        String output = outContent.toString().trim();

        

        assertTrue("Output harus mengandung informasi yang relevan", 

                   output.contains("Jumlah mesin: 10"));

        

        assertTrue("Output harus mengandung harga mesin yang benar", 

                   Double.parseDouble(output.split("Harga mesin: ")[1].split(" ")[0]) == 500000.0);

        

        double totalBiaya = Double.parseDouble(output.split("Total biaya produksi: ")[1]);

        assertEquals("Total Biaya Produksi harus benar", totalBiaya, (10 * 500000), 100); // 100 adalah batasan error

    }

}